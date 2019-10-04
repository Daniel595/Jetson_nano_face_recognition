#include "recognize.h"

recognize::recognize(void){

    deserialize("src/models/dlib_face_recognition_resnet_model_v1.dat") >> net; 
    deserialize("src/models/shape_predictor_5_face_landmarks.dat") >> sp;          
                         
}
    
recognize::~recognize(void){

}    

void recognize::init(){
    printf("initialization of recognition network!\n");
    matrix<rgb_pixel> img;
    matrix<float,0,1> face_descriptor;
    load_image(img, "res.jpg");
    this->embedding(&img, &face_descriptor);
    //check if test image matches
}

//get embeddings from a single cropped face
void recognize::embedding(matrix<rgb_pixel> *face_chip, matrix<float,0,1> *face_descriptor){
    
    *face_descriptor = net(*face_chip);
    
}

//get embeddings from a vector of cropped faces
void recognize::embeddings(std::vector<matrix<rgb_pixel>> *face_chips, std::vector<matrix<float,0,1>> *face_descriptors){

    *face_descriptors = net(*face_chips);
    //cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;
}



//print out the network architecture
void recognize::tell(){
    cout << "The net has " << net.num_layers << " layers in it." << endl;
    cout << net << endl;
}