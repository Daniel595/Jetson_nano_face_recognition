#include "face_classifier.h"   
   


    
face_classifier::face_classifier(face_embedder *embedder){
    this->embedder = *embedder;
    
    int svm_trained = 1;                            
    // check if labels exist
    svm_trained &= get_labels();
    std::vector<string> info;
    svm_trained &= get_info(&info);
    cout << "svms trained? : " << svm_trained << endl;

    if(svm_trained == 1){     
        int success = deserialize_svm(label_encoding.size());
        if(success == 0) train_svm(&info);
    }else{
        train_svm(&info);                                        
    }

    this->num_classes = label_encoding.size();
    this->num_classifiers = this->num_classes * (this->num_classes - 1) / 2;
    
    init();
}

face_classifier::~face_classifier(){

}



void face_classifier::init(){
    
    matrix<rgb_pixel> img;   
    dlib::load_image(img, test_dir);   // load test file
    this->embedder.embedding(&img, &test_embedding);
    
    double label;
    prediction(&test_embedding, &label);
    
    cout << "selftest classifier - label of test image is: " << label << " - " << label_encoding[label] << endl; 
       
}
    



void face_classifier::get_label_encoding(std::vector<std::string> *labels){
    *labels = this->label_encoding;
}



int face_classifier::get_labels(){
    string line;
    std::ifstream label_i;                          //stream to read labels
    int success = 1;
    
    label_i.open(labels_dir);
    if(label_i.is_open()){
        while(getline(label_i,line)) label_encoding.push_back(line);    //labels to array
        label_i.close();
    }else cout << "unable to open file" << endl;

    if(label_encoding.empty()) success &= 0;

    return success;
}


int face_classifier::get_info(std::vector<string> *info){
    
    std::ifstream info_i;                           //stream to read labels
    string line;
    int success = 1;

    info_i.open(info_dir); 
    if(info_i.is_open()) {
        while(getline(info_i,line)) info->push_back(line);       //info to array
        info_i.close();        
    } 
    else cout << "unable to open file" << endl;
    
    //if no info is there we want to train new svms 
    if(info->empty()){
        success &= 0;
        cout << "The file \"info.txt\" does not exist or is corrupted. Please run \"faces/generate_input_data.py\"." << endl;
    }  
    //if line 1 does not contain "+" a new training is required
    else if(info->at(0) != "+") success &= 0;

    return success;
}



// for single sample
void face_classifier::prediction(   sample_type_embedding *face_embedding, 
                                    double *face_label){
    

    std::vector<sample_type_embedding> face_embeddings;
    std::vector<double> face_labels;

    face_embeddings.push_back(*face_embedding);
    
    prediction(&face_embeddings, &face_labels);

    *face_label = face_labels[0];
}




// for  multiple samples
void face_classifier::prediction(   std::vector<sample_type_embedding> *face_embeddings, 
                                    std::vector<double> *face_labels){
    static double threshold = 0.0;
    sample_type_svm sample;

    // iterate all embeddings
    for(int i=0; i < face_embeddings->size(); i++ ){

        sample = matrix_cast<double>(face_embeddings->at(i));   // get next sample and cast it to required double format
        std::map<double, int> votes;                // map of class , num_of_votes
        double mean[this->num_classes] = {0};       // some kind of "confidence" per class
        
        // run every classifier
        for(int k = 0; k < classifiers.size(); k++){
            double prediction = classifiers[k](sample);
            //cout << prediction << " : ";
            
            if(abs(prediction) < threshold) {
                //cout << "-1" << endl;
            } else if (prediction < 0) {
                votes[classifiersLabels[k].first]++;                        // increment number of votes
                //cout << classifiersLabels[k].first << endl;
                mean[(int)classifiersLabels[k].first] += abs(prediction);   // add value to positive
                //mean[(int)classifiersLabels[k].second] -= abs(prediction);  // sub value from negative
            } else {
                votes[classifiersLabels[k].second]++;
                //cout << classifiersLabels[k].second << endl;
                mean[(int)classifiersLabels[k].second] += abs(prediction);
                //mean[(int)classifiersLabels[k].first] -= abs(prediction); 
            }
        }

        //cout << "Votes: " << endl;
        //for(auto &vote : votes) {
        //    cout << vote.first << ": " << vote.second << endl;
        //}


        double label = -1;
        double max = 0;
        int num_votes = 0;
        static int min_votes = this->num_classes/2 - 1;
        static double mean_threshold = 0.35;

        for(int i = 0; i<this->num_classes; i++){
            num_votes += votes[i];  
            if (votes[i] != 0){         // prevent Zero division
                mean[i] = (mean[i] < 0 ? 0 : mean[i]/votes[i]);    // claculate mean value
                if (mean[i] > max && votes[i] > min_votes){         // new maximum and at least 2 votes?
                    max = mean[i];                      
                    label = (max >= mean_threshold ? i : label);    // value above threshhold? -> new label
                }
            } 
            //printf("class: %d: mean: %f\n", i, mean[i] );
        }
        printf("label is %f\n",label);
        //printf("-1 votes: %d\n", this->num_classifiers - num_votes);

        face_labels->push_back(label);

    }
}
       




//read the training faces and prepare for training
void face_classifier::get_training_data(std::vector<sample_type_embedding> *face_embeddings, 
                                            std::vector<double> *labels, 
                                            std::vector<double> *total_labels ){

    using namespace boost::filesystem;
    path p(train_data_dir);                             //path where the cropped training data is located
    std::vector<matrix<rgb_pixel>> faces;               //all training sample images
    int num_labels = 0;                                 //number of labels
    
    try
    {
        if (exists(p))    
        {
        if (is_regular_file(p))        
            cout << p << "is a file. \nStart at directory \"parent\", when the following structure is given: \"parent/label/image.jpg\" " << endl;
        else if (is_directory(p))
        {
            recursive_directory_iterator dir(p), end;               //iterate over all directories and files (boost function)   
            while(dir != end){
                //current object is a new label (folder)
                if(is_directory(dir->path())) {    
                    label_encoding.push_back( dir->path().filename().string() );     // get label encoding
                    total_labels->push_back(num_labels);                            // store label number
                    num_labels ++;
                }
                //current object is a file
                else {                                 
                    matrix<rgb_pixel> img;                          
                    dlib::load_image(img, dir->path().string());    //load file
                    faces.push_back(img);                           //store face
                    labels->push_back(num_labels-1);                //store label of current files
                    cout << "label: " << num_labels - 1 << "   file: "<< dir->path().filename() << endl;
                }
                ++dir;
            }
        }
        else
            cout << p << " exists, but is neither a regular file nor a directory\n";
        }
        else
        cout << p << " does not exist\n";
        embedder.embeddings(&faces, face_embeddings);                   // create the training face embeddings
    }
    catch (const filesystem_error& ex)
    {
        cout << ex.what() << '\n';
    }
}



int face_classifier::deserialize_svm(int classes){
    
    int success = 1;
    
    cout << "try to deserialize SVMs:" << endl;
    //create svm names
    std::vector<string> svm_names;
    for(int i = 0; i<classes; i++){
        for(int k = i+1; k<classes; k++){
            string classifier_num = to_string(i) + "_" + to_string(k);
            string svm_name =  "svm/classifier_" + classifier_num + ".dat";
            svm_names.push_back(svm_name);
            classifiersLabels.emplace_back(make_pair((double)i, (double)k));
            cout << svm_name << endl;
        }
    }
    //deserialize svms
    for(int i = 0; i < svm_names.size(); i++ ){
        classifier_type cl;
        try{
            deserialize( svm_names[i] ) >> cl;    
            classifiers.emplace_back(cl);
        }catch(...){
            success &= 0;
            classifiers.clear();
            classifiersLabels.clear();
            cout << "Deserialisation of " << svm_names[i] << " failed. Start training SVMs." << endl;
        }
    }

    return success;
} 





void clear_svm_dir(string svm_dir){
    boost::filesystem::path svm_d (svm_dir);
    boost::filesystem::remove_all(svm_d);
    boost::filesystem::create_directory(svm_d);
}






void face_classifier::training(std::vector<sample_type_embedding> *face_embeddings, 
                                            std::vector<double> *labels, 
                                            std::vector<double> *total_labels ){

    //all training samples
    std::vector<sample_type_svm> samples;
    //cast all embeddings to svm format
    for(int i = 0; i < face_embeddings->size(); i++) samples.push_back(matrix_cast<double>( face_embeddings->at(i) )); 

    // Initialize trainers
    std::vector<trainer_type> trainers;
    int num_trainers = total_labels->size() * (total_labels->size()-1) / 2; 
    for(int i = 0; i < num_trainers ; i++) {
        trainers.emplace_back(trainer_type());
        trainers[i].set_kernel(kernel_type());
        trainers[i].set_c(10);
    }

    // train classifiers
    int label1 = 0, label2 = 1;

    for(trainer_type &trainer : trainers) {
        std::vector<sample_type_svm> samples4pair;
        std::vector<double> labels4pair;

        for(int i = 0; i < samples.size(); i++) {
            
            //check if the current embedding belongs to the positiv or negativ of the current SVM
            if(labels->at(i) == total_labels->at(label1)) {
                samples4pair.emplace_back(samples[i]);
                labels4pair.emplace_back(-1);
            }
            if(labels->at(i) == total_labels->at(label2)) {
                samples4pair.emplace_back(samples[i]);
                labels4pair.emplace_back(+1);
            }
        }
        randomize_samples(samples4pair, labels4pair);

        //training
        classifiers.emplace_back(trainer.train(samples4pair, labels4pair));         //train and store classifier
        classifiersLabels.emplace_back(make_pair(total_labels->at(label1),             //store the label relations of the classifier
                                                 total_labels->at(label2)));
        
        //serialize classifier
        string classifier_num = to_string((int)total_labels->at(label1)) + "_" + to_string((int)total_labels->at(label2));
        string classifier_name =  "svm/classifier_" + classifier_num + ".dat";
        serialize(classifier_name) << classifiers[classifiers.size()-1];
        cout << classifier_name << " generated!" << endl;
        //prepare for next iteration
        label2++;
        if(label2 == total_labels->size()) {
            label1++;
            label2 = label1 + 1;
        }
    }
}



 



void face_classifier::train_svm(std::vector<string> *info){
    
    clear_svm_dir(svm_dir);         // delete old svms
    label_encoding.clear();         // delete old encodings                             
    
    // get training data
    std::vector<sample_type_embedding> face_embeddings;             //all training face embeddings
    std::vector<double> labels;                                     //all training labels for the embeddings 
    std::vector<double> total_labels;                               //unique labels
    get_training_data(&face_embeddings, &labels, &total_labels);    // get training data
    
    // train svm and initialize the classifiers
    training(&face_embeddings, &labels, &total_labels);

    //write labels
    std::ofstream label_o(labels_dir);                  //stream to write to file
    for (int i = 0; i<label_encoding.size(); i++){ 
        label_o << label_encoding.at(i) << endl;        // write label to file
    }
    label_o.close();
    
    // write + to info-file
    std::ofstream info_o(info_dir);                     //stream to write to file
    if(info->empty()) info->push_back("+");
    else info->at(0) = "+";
    for(int i=0; i < info->size(); i++){
        info_o << info->at(i) << endl;
    }
    info_o.close();

}
 
