# Jetson nano face recognition

live with camera / or with images from disk

recognition example - trained to the main BBT-characters:
![alt text](https://github.com/Daniel595/testdata/blob/master/result/13.png)
more results at https://github.com/Daniel595/testdata/tree/master/result


## Parts:

1. detection: high performance MTCNN  (CUDA/TensorRT/C++). A fast C++ implementation of MTCNN, TensorRT & CUDA accelerated from https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT. MTCNN detects face locations wich will be cropped, aligned and fed into the "dlib_face_recognition_model".

2. recognition: dlib_face_recognition_model creates a 128-d face embedding for every input face. This will be used as SVM input for classification.

3. classification: svms will be trained based on the prepared dataset. We will have N*(N-1)/2 SVMs for N classes. Every input will be fed into every SVM and the "weight" of every class gets summed. For the summed values I use a threshold and he highest value above threshold wins.



## Information

1. facial images for training: 

        - location: faces/train/datasets/<set>/<class_name>/<images> 
                (see the bbt example)
    
        - preprocessing(prepare svm training): python3 faces/generate_train_data.py datasets/<set>   
                (after doing this the ./main will train the svms automatically)
    
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


4. training SVMs:
 
    A call of "python3 faces/generate_train_data.py datasets/<set>" generates training data. It will detect, crop, align and augment faces which will be used for training from ./main. 
    
    
    When calling "./main" the app detects if training data has changed since the last training. In this case it will train and store SVMs to "/svm" and asks for calling "./main" again. If training data has not changed the svms will be deserialized/read from "/svm"

    
    
5. run:
    
    Calling "./main" the first time the app will build TensorRT cuda engines for the MTCNN what takes about 3 mins. The engines will be serialized and reused. You can only feed images with the size the MTCNN was build for. Changing size will require new cuda engines for the first MTCNN-stage (P-net)
    

## Issues

Every class should have ~the same amount of training-images. If I put more images to one class (like every class 5 and one class 20) it always predicts this class.

In some rare cases the pipeline gets stuck after building the app partially. I didn't figured out why that happens but "make -B" and reboot do help here.


## Speed

No documented tests. It detects me with about ~30 FPS (one face) trained for 6 people. It is slowed down a lot by drawing bounding boxes and keypoints from CPU (TODO - from GPU). It still looks "fluent" at a detection of 5 persons. Adding more known faces will reduce the speed (because of more SVMs).

## TODO
```diff
- draw bounding and keypoints boxes from CUDA instead of CPU
- implement tracker to improve classification by considering the last predictions for the tracked face
- improve SVM evaluation
```
