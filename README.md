# Jetson nano face recognition

live with camera / or with images from disk

recognition example - trained to the main BBT-characters:
![alt text](https://github.com/Daniel595/BBT-faces/blob/master/test/result/96.png)


Parts:

detection: high performance MTCNN  (CUDA/TensorRT/C++)

recognition: dlib_face_recognition_model - using "face embeddings" similar to well known "FaceNet"

classification: dlib SVM's 


## MTCNN face detection

A fast C++ implementation of MTCNN, TensorRT & CUDA accelerated from https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT

## face recognition

1. dlib face recognition model to create face embeddings
    
    
2. dlib SVM's for classification
    


## Information

1. facial images for training: 

        - location: faces/train/datasets/<set>/<class_name>/<images> 
    
        - preprocessing: python3 faces/generate_train_data.py datasets/<set> - detect, extract, crop, align, prepare svm-training 
        
    
2. Required:

    C:

        - Dlib (cuda enabled)
    
        - Opencv (cuda enabled)
    
        - a built version of jetson-inference repo 
        https://github.com/dusty-nv/jetson-inference
    
        - /src/model/dlib_face_recognition_resnet_model_v1.dat
    
    
    
    Python: 
    
        - library face_recognition (https://pypi.org/project/face_recognition/)
    
        - Opencv
    
    
3. Build/run: 

        - add the training data, same num of pictures per face
        
        - python3 faces/generate_input_data.py (prepare training data)

        - cmake .
    
        - make
    
        - ./main


4. training:
 
    SVM's will be trained on startup by "face_classifier" if required. 
    
    The trained SVM's will be serialized to "/svm". 
    
    The "face_classifier" detects if the training-data changed since the last training. If so the SVM's will be trained again, otherwise the trained SVM's will be deserialized from "/svm".
    
    
5. run:
    
    At the first run the network will build cuda engines what takes about 3 mins. The engines will be serialized and reused. If the MTCNN input size changes the pnet engines need to be rebuilt because their inputsize depends on MTCNN iniputsize.
    

## Issues

Every class should have the same amount of training-images. If I put more images to one class (like every class 5 and one class 20) it always predicts this class. (wanna check SVM-basics - can i prevent this failure?) 


I found out that deserialized engines sometimes do not work after rebuilding some C++ modules. In this case it gets stuck during using the engine and you have to kill the process. It seems not to happen in case of a clean build. (??)

## Speed

Not really tested yet. The only things I can say is: it detects me with about 40 FPS (one face). It is slowed down a lot by drawing bounding boxes and keypoints from CPU (TODO - from GPU). It still looks "fluent" at a detection of about 5 persons.

## TODO
```diff
- draw bounding and keypoints boxes from CUDA instead of CPU
- implement tracker to improve classification by considering the last predictions for the tracked face
- SVM basics: SVM cross validation (param. optimization), different number of training samples
- Redesign "detections" as class, prepare for tracker implementation
```
