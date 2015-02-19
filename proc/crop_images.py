import os
import sys
sys.path.append('../lib/')
import util
import improc as ip
import cv2

reload(ip)

base_folder = '/Users/ivoeverts/data/wehkamp/';
image_folder = 'img/'
cropped_image_folder = 'img_cropped_test/'

img_list = util.list_files(os.path.join(base_folder,image_folder),'jpg')

ip.GDDImage.SCALE = 1
for i in range(0,len(img_list)):
	print img_list[i]
	im = ip.GDDImage(img_list[i],'I',True)
	cv2.imwrite(os.path.join(base_folder,cropped_image_folder,os.path.split(img_list[i])[1]),im.bgr_image)