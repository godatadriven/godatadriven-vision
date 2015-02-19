import os
import sys
sys.path.append('../lib/')
import improc as ip
import util
import pandas as pd

reload(ip)

do_build = True
if len(sys.argv) > 1:
	do_build = int(sys.argv[1])

base_folder = '/Users/ivoeverts/data/wehkamp/';
image_folder = 'img/'

if do_build:
	indexer = ip.VisualObjectMatcher({'max_num_db_images':10,'base_folder':base_folder,'image_folder':image_folder}, False)
	names = ['file_name','description','tax1','tax2','tax3']
	meta = pd.read_csv(os.path.join(base_folder,'meta/clothes.csv'), header=0, delimiter=',', dtype=object, names=names)
	filepaths = [os.path.join(base_folder,image_folder,f)+'.jpg' for f in meta.file_name]
	indexer.build_index(filepaths)

matcher = ip.VisualObjectMatcher({'base_folder':base_folder,'image_folder':image_folder})
img_list = util.list_files(os.path.join(base_folder,image_folder),matcher.image_extension)
for i in range(0,10):
	print matcher.match(img_list[i])