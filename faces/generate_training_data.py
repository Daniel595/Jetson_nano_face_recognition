import cv2
import os
import face_recognition_models
import numpy as np 
import os
import dlib
import shutil
import argparse
import random
import sys


#stolen from face_recognition api
def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def _augment(img, alpha, beta):
   return  cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# for augmentation:
# beta -100 to 100 brightness
brightness = [-25, +15, 0]
# alpha 1-3 contrast
contrast = [1, 1.3, 0.7]


#paths
raw_path = os.path.abspath(os.path.dirname(__file__))
train_src = os.path.abspath(sys.argv[1])
train_dst = os.path.join(raw_path, "train/cropped")
if(os.path.isdir(os.path.join(train_dst))):
    shutil.rmtree(train_dst)
os.mkdir(train_dst)
src = train_src
dst = train_dst
 
#models
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
shape_predictor = dlib.shape_predictor(predictor_68_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

label = "none"
cnt = 0
failed_paths = ""
fail_cnt = 1
num_labels = 0
num_label_images = []
labels = []

#for i in range(len(src)):
print("source directory: " + src)
print("destination directory: " + dst)
for path, dirs, files in os.walk(src):

    for f in files:
        if not f.endswith(".gitignore"):
            split = path.split("/")
            name = split[len(split)-1]
                
            if(name != label):
                if(cnt > 0):
                    num_label_images.append(cnt)
                cnt = 0
                print("new label: " + name)
                label = name
                labels.append(label)
                os.mkdir(os.path.join(dst, label))
                num_labels += 1
                
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
                    chip = cv2.medianBlur(chip, 3)
                    flip_chip = np.fliplr(chip)
                    prefix = "chip_" + label + "_"
                    number = ""
                    if(cnt < 10):
                        number = "0"
                    number = number + str(cnt)
                    filetype = f.split(".")
                    suffix = "." + filetype[len(filetype) - 1]
                    chip_name =   prefix + number + "_aug" 
                    aug = 0
                    for lev in brightness:
                        #dlib.save_image(chip, os.path.join(dst[i], label, chip_name + "0" + suffix ))
                        dlib.save_image(_augment(chip, random.choice(contrast), lev), os.path.join(dst, label, chip_name + str(aug) + suffix ))
                        dlib.save_image(_augment(flip_chip, random.choice(contrast), lev), os.path.join(dst, label, chip_name + str(aug+1) + suffix))
                        aug+=2

                    print("Face successfully extracted!\n")
            #new file
            cnt = cnt + 1    

num_label_images.append(cnt)

filename = "info.txt"
info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

try:
    os.remove(info_path)
except:
    print("failed accessing info.txt")

info_file = open(info_path, "w+")
info_file.write("training required!\n" + "num classes: " + str(num_labels) + "\n")
info_file.write("#\tlabel:\timages:\n" )

for i in range(len(num_label_images)):
    info_file.write(str(i) + "\t" +labels[i] +"\t" + str(num_label_images[i]) +  "\n")
info_file.close()

if(len(failed_paths) > 0):
    print("Face extraction from the following pictures didn't work: ")
    print(failed_paths)
else:
    print("All pictures successfully processed")    
            