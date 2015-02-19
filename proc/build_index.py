import os
import sys
sys.path.append('../lib/')
import improc as ip
import pandas as pd

reload(ip)

do_build = True
if len(sys.argv) > 3:
	do_build = int(sys.argv[4])
elif len(sys.argv) < 3:
	print "usage: python build_index.py FEAT COLOR [do_build]"

opts = {'base_folder':'/Users/ivoeverts/data/wehkamp/',
		'image_folder':'img_cropped/',
		'feat':sys.argv[1],
		'color':sys.argv[2]}

names = ['file_name','description','tax1','tax2','tax3']
meta = pd.read_csv(os.path.join(opts['base_folder'],'meta/clothes.csv'), header=0, delimiter=',', dtype=object, names=names)
filepaths = [os.path.join(opts['base_folder'],opts['image_folder'],f)+'.jpg' for f in meta.file_name]

if do_build:
	indexer = ip.VisualObjectMatcher(dict({'max_num_db_images':6000}.items()+opts.items()), False)
	indexer.build_index(filepaths)

matcher = ip.VisualObjectMatcher(opts)
for i in range(0,10):
	print matcher.match(filepaths[i])