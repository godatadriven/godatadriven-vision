# example
# python build_index ../data/ SURF I 4

import os
import sys
import pandas as pd
sys.path.append('../lib/')
import improc as ip
import util

# for dev purposes:
reload(ip)

# possibility to only run match example
do_build = True
if len(sys.argv) > 5:
	do_build = int(sys.argv[5])
elif len(sys.argv) < 5:
	print "usage: python build_index.py BASE_FOLDER FEAT COLOR K [do_build]"

# settings
opts = {'base_folder':sys.argv[1],
		'feat':sys.argv[2],		# SURF or HIST
		'color':sys.argv[3],	# I or r or g
		'k':int(sys.argv[4])}	# small (4) for simple images, large (32) for complex images

# meta data, wehkamp case
#names = ['file_name','description','tax1','tax2','tax3']
#meta = pd.read_csv(os.path.join(opts['base_folder'],'meta/clothes.csv'), header=0, delimiter=',', dtype=object, names=names)
#filepaths_ = [os.path.join(opts['base_folder'],'img/',f)+'.jpg' for f in meta.file_name]
#filepaths = [f for f in filepaths_ if os.path.exists(f)]

# filepaths, generic case
filepaths = util.list_files(os.path.join(opts['base_folder'],'img/'),'jpg')

# build index
if do_build:
	indexer = ip.VisualObjectMatcher(dict({'max_num_db_images':6000}.items()+opts.items()), False)
	indexer.build_index(filepaths)
else:
	# match example: best match should have a distance of 0
	matcher = ip.VisualObjectMatcher(opts)
	for i in range(0,10):
		print matcher.match(filepaths[i])