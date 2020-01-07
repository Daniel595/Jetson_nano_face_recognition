## Content
- bbt - contains testdata
- train/datasets/bbt - contains bbt dataset which is used for the testdata submodule https://github.com/Daniel595/testdata
- generate_training_data.py - script to crop, align, augment faces for svm training
- info.txt - used by python script and by ./main to detect if new there is training data since the last training and for an overview
- labels.txt - used by ./main during training, contains the face labels
- test.jpg - An image named test.jpg will be used to initialize the "face_embedder" and "face_classifier" 

## Preprocessing

- python3 generate_training_data.py "path/to/dataset"

This will crop, align and augment faces from the choosen dataset and store it to a path which is knwon in ./main. The Labels will be given by the name of the directories.

