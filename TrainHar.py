import urllib.request
import cv2
import numpy as np
import os
import argparse
import sys
from subprocess import call
import socket


# need negative images (many)
# 1-> search http://www.image-net.org/ for unrealted imgs and get url list. Busy background heavy images are best
# 2-> run this program with the above url as arg --imagenet_search_urls
# need positive images (images  that contain the object to identify) (fewer)
# 1-> manualy put positive image in 'pos' folder, next to 'neg'. about 40, with different light/background.
# 2-> manualy tag the ROI of the positive images.
# 3-> run the .bat file in the positive images folder.
# 3-> generate samples.

# --imagenet_search_urls
#
#  20 bugs:        http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02187554
#  2k mountains    http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09403734
# --output_directory imgnet

# http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html


def download_neg_images():
    socket.setdefaulttimeout(10)  # times out non responding fetchs when downloading images
    pic_num = 1
    img_dir = os.path.join(sys.path[0], "imgnet")
    img_dir = os.path.join(img_dir, 'neg')

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    f = open(os.path.join('imgnet/neg', 'links.txt'), 'r')
    imagenet_search_urls = f.read().splitlines()
    cnt = 1
    for imagenet_search_url in imagenet_search_urls:
        cnt=cnt+1
        try:
            print(str(cnt) + '/ ' + str(len(imagenet_search_urls)))
            urllib.request.urlretrieve(imagenet_search_url, os.path.join(img_dir, "tmp.jpg"))  # downloads image
            img = cv2.imread(os.path.join(img_dir, "tmp.jpg"), cv2.IMREAD_GRAYSCALE)
            # avoid really small images
            if img.size > 20000:
                # make the images 100px x ~relative px
                width, height = img.shape[1::-1]
                adj = 400 / width
                resized_image = cv2.resize(img, (400, int(round(height*adj))))
                # some sites return a "not found" placeholder image. need to avoid saving those.
                good = True
                for junk in os.listdir("imgnet/junk"):
                    junk_img = cv2.imread(os.path.join("imgnet/junk", junk))
                    if junk_img.shape == img.shape and not(np.bitwise_xor(junk_img, img).any()):
                        good = False
                if good:
                    cv2.imwrite(os.path.join(img_dir, str(pic_num) + ".jpg"), resized_image)
                    pic_num += 1
        except Exception as e:
            print(str(e))

    try:
        os.remove(os.path.join(img_dir, "tmp.jpg"))
    except OSError:
        pass
    generate_negative_file_list()


def generate_negative_file_list():
    img_dir = os.path.join(sys.path[0], "imgnet")
    img_dir = os.path.join(img_dir, 'neg')
    try:
        os.remove(os.path.join(img_dir, 'negatives.txt'))
    except OSError:
        pass
    # create_descriptors. do it here instead of in-line to ensure the text file is accurate.
    for img in os.listdir(img_dir):
        if img.endswith(".jpg") or img.endswith(".jpeg"):
            line = img + '\n'
            with open(os.path.join(img_dir, 'negatives.txt'), 'a') as f:
                f.write(line)


def generate_training_images():
    for img in os.listdir("imgnet/train"):
        if img.endswith(".jpg") or img.endswith(".jpeg"):
            print(img)
            # img = cv2.imread(os.path.join("imgnet/pos", img), cv2.IMREAD_GRAYSCALE)  # reads image

            call([
                  "imgnet\opencv_createsamples",
                  "-img", os.path.join(sys.path[0], os.path.join(os.path.join("imgnet", "train")), img),
                  "-bg", os.path.join(sys.path[0], os.path.join(os.path.join("imgnet", "neg")), "negatives.txt"),
                  "-info", os.path.join(sys.path[0], os.path.join(os.path.join("imgnet", "traincln")), "info.txt"),
                  "-num", "128",
                  "-maxxangle", "1.0",
                  "-maxyangle", "1.0",
                  "-maxzangle", "0.5",
                  "-bgcolor", "255",
                  "-bgthresh", "8",
                  "-w", "40",
                  "-h", "40"])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='neg, pos, train, negfiles', required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    arg = parse_arguments(sys.argv[1:])
    if arg.mode == "neg":
        download_neg_images()
    elif arg.mode == "pos":
        generate_training_images()
    elif arg.mode == "negfiles":
        generate_negative_file_list()
    else:
        print("todo")

