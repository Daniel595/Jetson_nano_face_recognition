//
// Created by zhou on 18-5-4.
//

#include "baseEngine.h"
baseEngine::baseEngine(const char * prototxt,const char* model,const  char* input_name,const char*location_name,
                       const char* prob_name, const char *point_name) :
                             prototxt(prototxt),
                             model(model),
                             INPUT_BLOB_NAME(input_name),
                             OUTPUT_LOCATION_NAME(location_name),
                             OUTPUT_PROB_NAME(prob_name),
                             OUTPUT_POINT_NAME(point_name)
{
};
baseEngine::~baseEngine() {
    shutdownProtobufLibrary();
}

void baseEngine::init(int row,int col) {

}
void baseEngine::caffeToGIEModel(const std::string &deployFile,                // name for caffe prototxt
                                  const std::string &modelFile,                // name for model
                                  const std::vector<std::string> &outputs,   // network outputs
                                  unsigned int maxBatchSize,                    // batch size - NB must be at least as large as the batch we want to run with)
                                  IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
    cout<<"build cuda engine for " << model << endl;
    
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              nvinfer1::DataType::kFLOAT);
    // specify which tensors are outputs
    for (auto &s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 28);
    ICudaEngine*engine = builder->buildCudaEngine(*network);
    assert(engine);
    context = engine->createExecutionContext();

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    builder->destroy();

}



void baseEngine::serialize_engine(const std::string &name){
    std::string store_file = engine_path + name;
    cout << "serialize engine: " << store_file  << endl;
    
    IHostMemory * serializedModel = context->getEngine().serialize();
    std::ofstream ofs(store_file.c_str(), std::ios::binary);
    ofs.write((char*)(serializedModel->data()), serializedModel->size());
    ofs.close();
    serializedModel->destroy();
}


bool baseEngine::deserialize_engine(const std::string &name){   
    std::string check_file = engine_path + name;
    cout << "try to deserialize engine: " << check_file << endl;
    
    std::vector<char> trtModelStream_;
    size_t size{ 0 };

    std::ifstream file(check_file.c_str(), std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream_.resize(size);
        file.read(trtModelStream_.data(), size);
        file.close();
    }else{
        cout << "engine does not exist!" << endl;
        return false;
    } 
    
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
    assert(engine);
    context = engine->createExecutionContext();

    if(!engine){
        printf("failed to deserialize engine\n");
        return false;
    }
    if( !context ){
		printf("failed to create execution context\n");
        return false;
	}
    
    return true;
}