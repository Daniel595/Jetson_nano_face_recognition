from PIL import Image
import shutil
import os

def fit_size(im, fill_color=(0, 0, 0, 0), new_x=1280, new_y=720):
    x, y = im.size
    if x == new_x and y == new_y:
        return im
    x_ratio = x/new_x
    y_ratio = y/new_y
    if x_ratio > y_ratio:
        factor = new_x/x
        im = im.resize((new_x, int(factor*y) ))
    else:
        factor = new_y/y
        im = im.resize((int(factor*x), new_y)) 
    # make black background
    new_im = Image.new('RGBA', (new_x, new_y), fill_color)
    # paste in new image to the middle
    new_im.paste(im)
    return new_im

# clear old test data
raw_path = os.path.abspath(os.path.dirname(__file__))
test_dst = os.path.join(raw_path, "testdata/test")
test_src = os.path.join(raw_path, "testdata/raw")
if(os.path.isdir(os.path.join(test_dst))):
        shutil.rmtree(test_dst)
        os.mkdir(test_dst)

# create test data
cnt = 0
for path, dirs, files in os.walk(test_src):
    for f in files:
        print("processing: " + f)
        image_name = raw_path + "/testdata/test/resized_" + str(cnt) + ".png"
        image_path = os.path.join(path,f)
        test_image = Image.open(image_path)
        new_image = fit_size(test_image) 
        new_image.save(image_name, 'PNG')  
        cnt += 1

result_dst = os.path.join(raw_path, "testdata/result")
if(os.path.isdir(os.path.join(result_dst))):
        shutil.rmtree(result_dst)
        os.mkdir(result_dst)