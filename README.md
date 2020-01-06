# Jetson nano face recognition

live with camera / or with images from disk

recognition example - trained to the main BBT-characters:
![alt text](https://github.com/Daniel595/testdata/blob/master/result/96.png)
more results at https://github.com/Daniel595/testdata/tree/master/result


## Parts:

1. detection: high performance MTCNN  (CUDA/TensorRT/C++). A fast C++ implementation of MTCNN, TensorRT & CUDA accelerated from https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT

2. recognition: dlib_face_recognition_model - using "face embeddings" 

3. classification: dlib SVM's 



## Information

1. facial images for training: 

        - location: faces/train/datasets/<set>/<class_name>/<images> 
    
        - preprocessing(prepare svm training): python3 faces/generate_train_data.py datasets/<set>   
        
    
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

        - add the training data, ~same num of pictures for each face
        
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

Every class should have the same amount of training-images. If I put more images to one class (like every class 5 and one class 20) it always predicts this class.


## Speed

No documented tests. It detects me with about ~30 FPS (one face) trained for 6 people. It is slowed down a lot by drawing bounding boxes and keypoints from CPU (TODO - from GPU). It still looks "fluent" at a detection of 5 persons. Adding more known faces will reduce the speed (because of more SVMs).

## TODO
```diff
- draw bounding and keypoints boxes from CUDA instead of CPU
- implement tracker to improve classification by considering the last predictions for the tracked face
- improve SVM evaluation
```
