This repo is inactive 

# Jetson nano face recognition


live with camera / or with images from disk

recognition example - trained to the main BBT-characters:
![alt text](https://github.com/Daniel595/testdata/blob/master/result/13.png)
more results at https://github.com/Daniel595/testdata/tree/master/result


## Parts:

1. detection: high performance MTCNN  (CUDA/TensorRT/C++). A fast C++ implementation of MTCNN, TensorRT & CUDA accelerated from [here](https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT). I adapted the to the Jetson Nano. MTCNN detects face locations wich will be cropped, aligned and fed into the "dlib_face_recognition_model". 

2. recognition: dlib_face_recognition_model creates a 128-d face embedding for every input face. This will be used as SVM input for classification.

3. classification: svms will be trained based on the prepared dataset. We will have N*(N-1)/2 SVMs for N classes. Every input will be fed into every SVM and the "weight" of every class gets summed. For the summed values I use a threshold and he highest value above threshold wins.


## Dependencies
I Recommend 64GB SD if you want to build OpenCV/Dlib

- I set my nano up as described [here](https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd) (including building Dlib from source)
- cuda enabled Dlib (Python and C, 19.17.0 tested)
- cuda enabled Opencv (Python and C, 4.1.0 tested, [link](https://github.com/mdegans/nano_build_opencv) )
- a built version of jetson-inference repo ([link](https://github.com/dusty-nv/jetson-inference))
- install python lib face_recognition ([link](https://pypi.org/project/face_recognition/), "pip install face_recognition")
- git clone https://github.com/Daniel595/Jetson_nano_face_recognition.git (without bbt testdata)
        or with bbt testdata:        
        git clone --recurse-submodules -j8 https://github.com/Daniel595/Jetson_nano_face_recognition.git
- Download [dlib_face_recognition_resnet_model_v1.dat](https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2) to "src/model/"
- replace the path dependencies in CMakeList with the paths on your System
- make sure the link "src/includes/" points to the includes-dir of your built "jetson-inference" repo


## Build/Run

- prepare training data 
- build: "cmake ."
- build: "make -j"
- run: "./main" 
- run: "./main" ( required a second time if svms were trained )


### train face classifier (SVM) 
Training happens automatically after adding and preprocessing training data.

- data location: faces/train/datasets/"set"/"class_name"/"img"  (.jpg, .png, see the [bbt-example](https://github.com/Daniel595/Jetson_nano_face_recognition/tree/master/faces/train/datasets/bbt))    
        
- data preprocessing: "python3 faces/generate_train_data.py datasets/"set""   
                (after doing this the ./main will train the svms automatically)
    

"python3 faces/generate_train_data.py datasets/<set>" will detect, crop, align and augment faces which will be stored and used for training. When calling "./main" the app detects if the training data has changed since the last training. In this case it will train and store SVMs to "/svm" and ask for calling "./main" again. If training data has not changed the svms will be deserialized from "/svm".


### Generate TensorRT MTCNN

Calling "./main" the first time the app will build TensorRT cuda engines for the MTCNN (approx. 3 mins). The engines will be serialized and reused in "engines/". 

You can only feed images with the size the MTCNN was build for. That means images with different size can't be fed or need to be resized. Changing the size will require new cuda engines for the first MTCNN-stage (P-net).
    
    

## Issues

Sometimes the MTCNN-pipeline gets stuck after building the project partially. I didn't figure out why that happens. But building the entire project ("make -B") and reboot does fix the problem. To avoid this you could always run "make -B" what takes longer than just make.


## Speed

In Camera-mode it detects me with about 30 FPS (one face) trained on 6 classes. It is slowed down a lot by drawing bounding boxes and keypoints from CPU (TODO - do it from GPU). Adding more known classes will reduce the speed because of more SVMs.

To test FPS I read in a picture with some faces and "loop over" it. I create copy of the image at the begin of every iteration to approx. simulate the image capturing by camera, feed it to the Pipeline and show the result. For 10 faces at image and 7 classes trained it runs at about 10 FPS.

![alt text](https://github.com/Daniel595/Jetson_nano_face_recognition/blob/master/pictures/fps/result_10.png)




## TODO
```diff
- setup
- make non rectangular Bboxes rectangular to keep face ratio (face partially out of the camera range (rare))
- draw bounding box and keypoints from CUDA instead of CPU
- implement object tracker to improve classification by considering recent predictions for the tracked face
- improve SVM evaluation (kernel, cross validation, parameters)
```
