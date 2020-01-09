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


test_image = Image.open("bbt_10_o.jpg")
new_image = fit_size(test_image) 
new_image.save("bbt_10.png", 'PNG')  

