//Author : Daniel595

#include "face_classifier.h"   
   


    
face_classifier::face_classifier(face_embedder *embedder){

    int svm_trained = 1;     
    this->restart = 0;                      
    // check if labels exist
    svm_trained &= get_labels();
    std::vector<string> info;
    svm_trained &= get_info(&info);

    if(svm_trained == 1){     
        int success = deserialize_svm(label_encoding.size());
        if(success == 0) train_svm(&info, embedder);
    }else{
        train_svm(&info, embedder);                                        
    }

    this->num_classes = label_encoding.size();
    this->num_classifiers = this->num_classes * (this->num_classes - 1) / 2;
    
    init(embedder);
}

face_classifier::~face_classifier(){

}

int face_classifier::need_restart() {
    return restart;
}

// make first prediction 
void face_classifier::init(face_embedder *embedder){
    
    matrix<rgb_pixel> img;   
    dlib::load_image(img, test_dir);   // load test file
    embedder->embedding(&img, &test_embedding);
    
    double label;
    prediction(&test_embedding, &label);
    
    cout << "selftest classifier - label of test image is: " << label << " - " << label_encoding[label] << endl; 
    // prediction OK?   
}



// get the labels of the classes / the persons names
void face_classifier::get_label_encoding(std::vector<std::string> *labels){
    *labels = this->label_encoding;
}


// read the labels from .txt file 
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

// read info.txt and check if training data has changed. If data changes we need to retrain SVM's.
// returns 0 if training is required, else 1 
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



// prediction for single sample
void face_classifier::prediction(   sample_type_embedding *face_embedding, 
                                    double *face_label){
    

    std::vector<sample_type_embedding> face_embeddings;
    std::vector<double> face_labels;

    face_embeddings.push_back(*face_embedding);
    
    prediction(&face_embeddings, &face_labels);

    *face_label = face_labels[0];
}




// prediction for multiple samples. 
//  - Iterate over all samples  
//  - Run all SVM's on every sample
//  - predict label per sample
//  - generate a vector of labels in the same sequence as the samples (face_labels)
//  
void face_classifier::prediction(   std::vector<sample_type_embedding> *face_embeddings, 
                                    std::vector<double> *face_labels){
    static double threshold = 0.0;
    sample_type_svm sample;

    // iterate all embeddings
    for(int i=0; i < face_embeddings->size(); i++ ){

        sample = matrix_cast<double>(face_embeddings->at(i));   // get next sample and cast it to required double format
        std::map<double, int> votes;                // map of class , num_of_votes
        double summed[this->num_classes] = {0};
        
        // run every classifier
        for(int k = 0; k < classifiers.size(); k++){
            double prediction = classifiers[k](sample);
            
            
            if(abs(prediction) < threshold) {
                //cout << "N";
            } else if (prediction < 0) {
                votes[classifiersLabels[k].first]++;                        // increment number of votes
                //cout << classifiersLabels[k].first ;
                summed[(int)classifiersLabels[k].first] += abs(prediction);   // add value to positive
                //summed[(int)classifiersLabels[k].second] -= abs(prediction);  // sub value from negative
            } else {
                votes[classifiersLabels[k].second]++;
                //cout << classifiersLabels[k].second ;
                summed[(int)classifiersLabels[k].second] += abs(prediction);
                //summed[(int)classifiersLabels[k].first] -= abs(prediction); 
            }
            //cout << " : " << prediction << endl;
        }

        
        //cout << "Votes: " << endl;
        //for(auto &vote : votes) {
        //    cout << vote.first << ": " << vote.second << endl;
        //}

        // Classify by mean value
        double label = -1;
        double max = 0;
        double mean = 0;
        static int min_votes = this->num_classes/2 - 1;
        static double mean_threshold = 0.6;
        static double sum_threshold = 2.5;

        // check for highest mean value
        for(int i = 0; i<this->num_classes; i++){
            if (votes[i] != 0){         // prevent Zero division
                mean = (summed[i] < sum_threshold ? 0 : summed[i]/votes[i]);    // claculate mean value
                if (mean>max && votes[i]>min_votes){         // new maximum and at least min_votes?
                    max = mean;                      
                    label = (max >= mean_threshold ? i : label);    // set label
                }
            }
            printf("class: %d:\tmean: %f\tvotes: %d\n\t\tsumm: %f\n", i, mean, votes[i],summed[i]);
        }
        printf("label: %f\n\n",label);
        //printf("-1 votes: %d\n", this->num_classifiers - num_votes);

        face_labels->push_back(label);

    }
}
       




// get training data from filesystem
//  - check the training data location
//  - get all images from every class and generate traing data (store to "face_embeddings")
//  - store all unique labels to "total_labels" and the labels sequence of the training data to "labels" 
void face_classifier::get_training_data(std::vector<sample_type_embedding> *face_embeddings, 
                                            std::vector<double> *labels, 
                                            std::vector<double> *total_labels, 
                                            face_embedder *embedder){

    using namespace boost::filesystem;
    path p(train_data_dir);                             //path where the cropped training data is located
    std::vector<matrix<rgb_pixel>> faces;               //all training sample images
    int num_labels = 0;                                 //number of labels
    
    try
    {
        // check path
        if (exists(p))    
        {
            // check entry point
            if (is_regular_file(p))
                cout << p << "is a file. \nStart at directory \"parent\", when the following structure is given: \"parent/label/image.jpg\" " << endl;
            else if (is_directory(p))   
            {

                // iterate over all directories and files - boost function diggs into a folder and then gets all files from inside 
                recursive_directory_iterator dir(p), end;                 
                while(dir != end){
                    
                    // current object is a folder - that means we have a new label.
                    if(is_directory(dir->path())) {    
                        label_encoding.push_back( dir->path().filename().string() );     // get label encoding
                        total_labels->push_back(num_labels);                            // store label number
                        num_labels ++;
                    }
                    
                    //current object is a file - that means we have another training example of the current label
                    else {                                 
                        matrix<rgb_pixel> img;                          
                        dlib::load_image(img, dir->path().string());    //load file
                        faces.push_back(img);                           //store face
                        labels->push_back(num_labels-1);                //store label of face
                        cout << "label: " << num_labels - 1 << "   file: "<< dir->path().filename() << endl;
                    }
                    ++dir;
                }
            } else cout << p << " exists, but is neither a regular file nor a directory\n";     // catch
        } else cout << p << " does not exist\n";                                                // catch
        
        // create the actual training data, the face embeddings
        embedder->embeddings(&faces, face_embeddings);                   
    }
    catch (const filesystem_error& ex)      // catch
    {
        cout << ex.what() << '\n';
    }
}


// get trained SVM's from disk
// returns 0 if at least one classifier couldn't be deserialized, else 1
int face_classifier::deserialize_svm(int classes){
    
    int success = 1;
    
    cout << "try to deserialize SVMs:" << endl;
    
    //create svm names "all vs. all" manner - thats how the svms are stored
    std::vector<string> svm_names;
    for(int i = 0; i<classes; i++){
        for(int k = i+1; k<classes; k++){
            // class combination
            string classifier_num = to_string(i) + "_" + to_string(k);
            // full name
            string svm_name =  "svm/classifier_" + classifier_num + ".dat";
            svm_names.push_back(svm_name);
            // store the two classes of this classifier - for prediction (which class is used in which classifier)
            classifiersLabels.emplace_back(make_pair((double)i, (double)k));
            cout << svm_name << endl;
        }
    }

    // deserialize svms by the previously generated names
    // iterate all names
    for(int i = 0; i < svm_names.size(); i++ ){
        classifier_type cl;
        
        // get the svm from disk
        try{
            deserialize( svm_names[i] ) >> cl;    
            classifiers.emplace_back(cl);
        // if deserialization of one classifier fails -> training is required
        }catch(...){
            success &= 0;               // fail-flag
            classifiers.clear();        // delete all serialized classifiers
            classifiersLabels.clear();  // delete all generated labels
            cout << "Deserialisation of " << svm_names[i] << " failed. Start training SVMs." << endl;
            return success;
        }
    }

    return success;
} 



// delete all trained svm's from filesystem
void clear_svm_dir(string svm_dir){
    boost::filesystem::path svm_d (svm_dir);
    boost::filesystem::remove_all(svm_d);
    boost::filesystem::create_directory(svm_d);
}




// perform SVM training
//  - train and serialize: N*(N-1)/2   SVM's (all vs. all)
void face_classifier::training(std::vector<sample_type_embedding> *face_embeddings, 
                                            std::vector<double> *labels, 
                                            std::vector<double> *total_labels ){
    //all training samples
    std::vector<sample_type_svm> samples;
    //cast all embeddings to required format
    for(int i = 0; i < face_embeddings->size(); i++) samples.push_back(matrix_cast<double>( face_embeddings->at(i) )); 

    // Initialize trainers one per SVM
    // TODO: change Parameters of SVM? Cross-Validation? what can i do to improve SVM prediction 
    std::vector<trainer_type> trainers;
    int num_trainers = total_labels->size() * (total_labels->size()-1) / 2; 
    for(int i = 0; i < num_trainers ; i++) {
        trainers.emplace_back(trainer_type());
        trainers[i].set_kernel(kernel_type());
        trainers[i].set_c(5);
    }

    // 
    int label1 = 0, label2 = 1;
    
    // iterate over all trainers
    for(trainer_type &trainer : trainers) {
        std::vector<sample_type_svm> samples4pair;
        std::vector<double> labels4pair;

        // for every trainer, iterate over all samples (training embeddings)
        for(int i = 0; i < samples.size(); i++) {
            // check if the current embedding matches a class used in this trainer
            // match - positive of SVM
            if(labels->at(i) == total_labels->at(label1)) {
                samples4pair.emplace_back(samples[i]);
                labels4pair.emplace_back(-1);
            }
            // match - negative of SVM
            if(labels->at(i) == total_labels->at(label2)) {
                samples4pair.emplace_back(samples[i]);
                labels4pair.emplace_back(+1);
            }
        }
        randomize_samples(samples4pair, labels4pair);

        // here the actual training happens
        classifiers.emplace_back(trainer.train(samples4pair, labels4pair));             // train and get classifier 
        classifiersLabels.emplace_back(make_pair(total_labels->at(label1),              // get the labels of the classifier
                                                 total_labels->at(label2)));
        
        //serialize classifier (store to filesystem)
        string classifier_num = to_string((int)total_labels->at(label1)) + "_" + to_string((int)total_labels->at(label2));
        string classifier_name =  "svm/classifier_" + classifier_num + ".dat";
        serialize(classifier_name) << classifiers[classifiers.size()-1];
        cout << classifier_name << " generated!" << endl;
        
        //preparation for next iteration (generate next "all vs. all" label combination)
        label2++;
        if(label2 == total_labels->size()) {
            label1++;
            label2 = label1 + 1;
        }
    }
}



 
// combine training functions
//  - get training data
//  - initialize training of SVM's
//  - write write labels to labels.txt for later use
//  - write a flag to info.txt which says that no training is required until dataset changes
void face_classifier::train_svm(std::vector<string> *info, face_embedder *embedder){
    
    clear_svm_dir(svm_dir);         // delete old svms
    label_encoding.clear();         // delete old encodings                             
    
    // get training data
    std::vector<sample_type_embedding> face_embeddings;             // all training face embeddings (per training image)
    std::vector<double> labels;                                     // all training labels for the embeddings (per training image)
    std::vector<double> total_labels;                               // unique labels/possible labels
    get_training_data(&face_embeddings, &labels, &total_labels, embedder);    // get training data
    
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

    this->restart = 1;      // set flag 
    cout << "train: " << this->restart << endl;
}
 
