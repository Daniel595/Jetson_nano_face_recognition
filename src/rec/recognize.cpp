#include "recognize.h"

recognize::recognize(void){

    deserialize("src/models/dlib_face_recognition_resnet_model_v1.dat") >> net; 
    deserialize("src/models/shape_predictor_5_face_landmarks.dat") >> sp;          
                         
}
    
recognize::~recognize(void){

}    

int recognize::embeddings(matrix<rgb_pixel> *face_chip){

    matrix<float,0,1> face_descriptor = net(*face_chip);
    cout << "face descriptor for one face: " << trans(face_descriptor) << endl;

    resizable_tensor temp;
    net.to_tensor(face_chip, face_chip+1, temp);
    //cout << temp.data() << endl;
    
    return 0;
}

//rgb_pixel is struct with unsigned char red, blue green
int recognize::embeddings(std::vector<matrix<rgb_pixel>> *face_chips){
    
    //faces = vector with 150 x 150 face chips
    std::vector<matrix<float,0,1>> face_descriptors = net(*face_chips);
    cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;
    
    return 0;
}




void recognize::tell(){
    cout << "The net has " << net.num_layers << " layers in it." << endl;
    cout << net << endl;
      
   //resizable_tensor tensorius = layer<131>(net);

}