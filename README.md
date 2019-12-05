face recognition on jetson nano - detection/recognition/classification

# MTCNN face detection

A fast C++ implementation of TensorRT and CUDA accelerated MTCNN from https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT

# face recognition

1. dlib face recognition model to create face embeddings
    
    
2. dlib SVM's for classification
    


## Information

1. face images: 

    image location: faces/train/raw/<class_name>/<images> (I used 5 images per class)
    
    preprocessing: faces/generate_input_data.py - detect, extract, crop, align faces, prepare for svm-training
    
    testdata: not used yet
    
2. Required:

    C:

    Dlib (cuda)
    
    Opencv (cuda)
    
    jetson-inference/includes (/src)
    
    /src/model/dlib_face_recognition_resnet_model_v1.dat
    
    
    
    Python: 
    
    face_recognition (face_recognition_models)
    
    Opencv
    
    
3. Build/run: 

    cmake .
    
    make
    
    ./main


4. training:
 
    SVM's will be trained on startup by "face_classifier" if required. 
    
    The trained SVM's will be "serialized" to "/svm". 
    
    The "face_classifier" detects if the training-data changed since the last training. If so the SVM's will be trained again, otherwise the trained SVM's will be deserialized from "/svm".
    


## Tests

classification works OK if every class has about the same amount of images. If i put lot more images to one class it always predicts this class. TODO: check ML-basics - doing a common failure? 

## Speed

about 40 FPS at one face. Slowed down a lot by drawing bounding boxes and keypoints by CPU.

## TODO
1. Serialize and deserialize TRT models for MTCNN, building them takes always about 3 mins
2. implement face tracker, improve classification by considering the last predictions for tracked face
3. SVM cross validation - optimize parameters for training
4. new design for detections - as class
