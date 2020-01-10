# Jetson nano face recognition

live with camera / or with images from disk

recognition example - trained to the main BBT-characters:
![alt text](https://github.com/Daniel595/testdata/blob/master/result/13.png)
more results at https://github.com/Daniel595/testdata/tree/master/result


## Parts:

1. detection: high performance MTCNN  (CUDA/TensorRT/C++). A fast C++ implementation of MTCNN, TensorRT & CUDA accelerated from https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT. I adapted the to the Jetson Nano. MTCNN detects face locations wich will be cropped, aligned and fed into the "dlib_face_recognition_model". 

2. recognition: dlib_face_recognition_model creates a 128-d face embedding for every input face. This will be used as SVM input for classification.

3. classification: svms will be trained based on the prepared dataset. We will have N*(N-1)/2 SVMs for N classes. Every input will be fed into every SVM and the "weight" of every class gets summed. For the summed values I use a threshold and he highest value above threshold wins.


## Dependencies
I Recommend 64GB SD if you want to build OpenCV/Dlib

- I set my nano up as described here https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd (including building Dlib from source)
- cuda enabled Dlib (Python and C, 19.17.0 tested)
- cuda enabled Opencv (Python and C, 4.1.0 tested, https://github.com/mdegans/nano_build_opencv) 
- a built version of jetson-inference repo (see: https://github.com/dusty-nv/jetson-inference)
- install python lib face_recognition (https://pypi.org/project/face_recognition/, "pip install face_recognition")
- git clone https://github.com/Daniel595/Jetson_nano_face_recognition.git (without bbt testdata)
        or with bbt testdata:        
        git clone --recurse-submodules -j8 https://github.com/Daniel595/Jetson_nano_face_recognition.git
- Download dlib_face_recognition_resnet_model_v1.dat to "src/model/" (https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2)
- replace the path dependencies in CMakeList with the paths on your System
- make sure the link "src/includes/" points to the includes-dir of your built "jetson-inference" repo


## Build/Run

- add training data (~same num of pictures for each class)  
- prepare training data: "python3 faces/generate_input_data.py train/datasets/bbt" (can be the path to any dataset) 
- "cmake ."
- build project: "make -j"
- run svm training: "./main" (training required only at first run after generating training data)
- run: "./main" (if SVMs were trained)


## train face classifier (SVM) 

Trainingdata:

- supposed location: faces/train/datasets/"set"/"class_name"/"images.x"  (tested: .jpg, .png, see the bbt example)    
        
- preprocessing: "python3 faces/generate_train_data.py datasets/"set""   
                (after doing this the ./main will train the svms automatically)
    
Training:

"python3 faces/generate_train_data.py datasets/<set>" will detect, crop, align and augment faces which will be stored and used for training. When calling "./main" the app detects if training data has changed since the last training. In this case it will train and store SVMs to "/svm" and asks for calling "./main" again. If training data has not changed the svms will be deserialized/read from "/svm".

    
## Generate TensorRT MTCNN

Calling "./main" the first time the app will build TensorRT cuda engines for the MTCNN what takes ~3 mins. The engines will be serialized and reused in dir "engines/". You can only feed images with the size the MTCNN was build for. Changing size will require new cuda engines for the first MTCNN-stage (P-net)


## Issues

Sometimes the MTCNN-pipeline gets stuck after building the project partially. I didn't figure out why that happens. But building the entire project ("make -B") and reboot does fix the problem. To avoid this you could always run "make -B" what takes longer than just make.


## Speed

In Camera-mode it detects me with about ~30 FPS (one face) trained on 6 classes. It is slowed down a lot by drawing bounding boxes and keypoints from CPU (TODO - do it from GPU). Adding more known classes will reduce the speed because of more SVMs.

To test FPS I read in a picture with a known amount of faces and loop over it. I create copy of the image at the begin of every iteration to simulate approximately the capturing by camera, feed it to the Pipeline and show the result. For 10 Faces I got about 10 FPS.

![alt text](https://github.com/Daniel595/Jetson_nano_face_recognition/blob/master/pictures/fps/result_10.png)




## TODO
```diff
- Dependencies: setup for all dependencies
- make non rectangular Bboxes (partially out of view faces (rare)) rectangular befor resizing to keep ratio 
- draw bounding and keypoints boxes from CUDA instead of CPU
- implement tracker to improve classification by considering the last predictions for the tracked face
- improve SVM evaluation
```
