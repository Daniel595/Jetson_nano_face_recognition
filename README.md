face recognition on jetson nano - detection/recognition/classification

# MTCNN face detection

A fast C++ implementation of TensorRT and CUDA accelerated MTCNN from https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT

# face recognition

1. dlib face recognition model to create face embeddings
    
    
2. dlib SVM's for classification
    


## Information

1. about facial images: 

        - location: faces/train/raw/<class_name>/<images> 
    
        - preprocessing: faces/generate_input_data.py - detect, extract, crop, align, prepare svm-training 
        
    
2. Required:

    C:

        - Dlib (cuda enabled)
    
        - Opencv (cuda enabled)
    
        - jetson-inference/includes (a built version of jetson-inference repo https://github.com/dusty-nv/jetson-inference)
    
        - /src/model/dlib_face_recognition_resnet_model_v1.dat
    
    
    
    Python: 
    
        - library face_recognition (https://pypi.org/project/face_recognition/)
    
        - Opencv
    
    
3. Build/run: 

        - add train data
        
        - python faces/generate_input_data.py

        - cmake .
    
        - make
    
        - ./main


4. training:
 
    SVM's will be trained on startup by "face_classifier" if required. 
    
    The trained SVM's will be serialized to "/svm". 
    
    The "face_classifier" detects if the training-data changed since the last training. If so the SVM's will be trained again, otherwise the trained SVM's will be deserialized from "/svm".
    
    
5. run:
    
    At the first run the network will build cuda engines what takes about 3 mins. The engines will be serialized and reused. If the MTCNN input size changes the pnet engines need to be rebuilt because their inputsize depends on MTCNN iniputsize.
    
6. Issues: 

    I found out that the engines sometimes don't work if you rebuild the C++ app. In this case it gets stuck in MTCNN face detection.  


## Tests

classification - OK if every class has about the same amount of images. If I put more images to one class (like every class 5 and one class 20) it always predicts this class. TODO: ML-basics - doing a common failure? 

## Speed

about 40 FPS at one face. Slowed down a lot by drawing bounding boxes and keypoints by CPU. TODO: draw bounding boxes by CUDA

## TODO
```diff
- initial testrun of all instances
- implement face tracker, improve classification by considering the last predictions for tracked face
- SVM cross validation - optimize parameters for training
- new design for detections - as class
```
