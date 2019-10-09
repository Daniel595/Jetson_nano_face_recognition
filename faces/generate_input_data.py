import cv2
import os
import face_recognition_models
import numpy as np 
import os
import dlib
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', help='pass \"--test=1\" if testpictures should be processed (default 0, not processing)', type=int, default=0)
parser.add_argument('--train',help='pass \"--train=0\" if trainpicture should not be processed (default 1, processing)', type=int, default=1)
args = parser.parse_args()

#stolen from face_recognition api
def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)



#paths
src = [] 
dst = []

raw_path = os.path.abspath(os.path.dirname(__file__))
if(args.test == 1):
    test_src = os.path.join(raw_path, "test/raw")
    test_dst = os.path.join(raw_path, "test/cropped")
    if(os.path.isdir(os.path.join(test_dst))):
        shutil.rmtree(test_dst)
    os.mkdir(test_dst)
    src.append(test_src)
    dst.append(test_dst)

if(args.train == 1):
    train_src = os.path.join(raw_path, "train/raw")
    train_dst = os.path.join(raw_path, "train/cropped")
    if(os.path.isdir(os.path.join(train_dst))):
        shutil.rmtree(train_dst)
    os.mkdir(train_dst)
    src.append(train_src)
    dst.append(train_dst)
 

#models
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
shape_predictor = dlib.shape_predictor(predictor_68_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

label = "none"
cnt = 0
failed_paths = ""
fail_cnt = 1

for i in range(len(src)):
    print("source directory: " + src[i])
    print("destination directory: " + dst[i])
    for path, dirs, files in os.walk(src[i]):

        for f in files:
            split = path.split("/")
            name = split[len(split)-1]
            
            if(name != label):
                cnt = 0
                print("new label: " + name)
                label = name
                os.mkdir(os.path.join(dst[i], label))
            
            image = dlib.load_rgb_image(os.path.join(path,f))
            print("processing " + f + " ..." + " (" + label + ")")

            #size image down
            div = (max(image.shape[0], image.shape[1]) / 800)
            new_wid = (int)(image.shape[0]/div)
            new_height = (int)(image.shape[1]/div)
            image = dlib.resize_image(image,new_wid, new_height)
            
            dets = face_detector(image, 1)
            
            if(len(dets) == 0):
                print("#### Failed to extract face: no face detected ####\n")
                failed_paths = failed_paths + str(fail_cnt) + ". " + os.path.join(path,f) + " -- no face detected. \n"
                fail_cnt = fail_cnt + 1
            elif(len(dets) > 1):
                print("#### Failed to extract face: too many faces in this picture ####\n")   
                failed_paths = failed_paths + str(fail_cnt) + ". " +  os.path.join(path,f) + " -- more than one face detected. \n"
                fail_cnt = fail_cnt + 1
            else:            
                for face in dets:
                    rect = _css_to_rect(_trim_css_to_bounds(_rect_to_css(face.rect), image.shape))
                    shape = shape_predictor(image, rect)
                    #face_chip extracts the face in a way that is expected as input by the recognition model
                    chip = dlib.get_face_chip(image,shape)

                    prefix = "chip_" + label + "_"
                    number = ""
                    if(cnt < 10):
                        number = "0"
                    number = number + str(cnt)
                    filetype = f.split(".")
                    suffix = "." + filetype[len(filetype) - 1]
                    chip_name =   prefix + number + suffix
                    dlib.save_image(chip, os.path.join(dst[i], label, chip_name))
                    print("Face successfully extracted!\n")
            #new file
            cnt = cnt + 1    


if(len(failed_paths) > 0):
    print("Face extraction from the following pictures didn't work: ")
    print(failed_paths)
else:
    print("All pictures successfully processed")    
            
