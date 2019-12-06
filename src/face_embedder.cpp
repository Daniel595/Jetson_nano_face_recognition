//Author : Daniel595

#include "face_embedder.h"

// load dlib embedding model
face_embedder::face_embedder(void){

    deserialize("src/models/dlib_face_recognition_resnet_model_v1.dat") >> net; 
    //deserialize("src/models/shape_predictor_5_face_landmarks.dat") >> sp;  
    init();        
                         
}
    
face_embedder::~face_embedder(void){

}    

// create a first embedding 
void face_embedder::init(){
    printf("initialization of embedding network!\n");
    matrix<rgb_pixel> img;
    matrix<float,0,1> face_embedding;
    load_image(img, "faces/test.jpg");
    this->embedding(&img, &face_embedding);
}

// get embeddings from a single cropped face
void face_embedder::embedding(matrix<rgb_pixel> *face_chip, matrix<float,0,1> *face_embedding){
    
    *face_embedding = net(*face_chip);
    cout << "face embedding for one face: " << trans(*face_embedding) << endl;
    
}

// get embeddings from a vector of cropped faces
void face_embedder::embeddings(std::vector<matrix<rgb_pixel>> *face_chips, std::vector<matrix<float,0,1>> *face_embeddings){

    *face_embeddings = net(*face_chips);
    //std::vector<matrix<float,0,1>> to_print = *face_embeddings;
    //cout << "face embedding for one face: " << to_print[0] << endl;
    //cout << "face embedding for one face: " << trans(to_print[0]) << endl;
    
}


// print out the network architecture
void face_embedder::tell(){
    cout << "The net consists of " << net.num_layers << " layers." << endl;
    cout << net << endl;
}