import os
import sys
sys.path.append('../lib/')
import improc as ip
import pandas as pd

# for dev purposes:
reload(ip)

# possibility to only run match example
do_build = True
if len(sys.argv) > 3:
	do_build = int(sys.argv[3])
elif len(sys.argv) < 3:
	print "usage: python build_index.py FEAT COLOR [do_build]"

# settings
opts = {'base_folder':'/Users/ivoeverts/data/wehkamp/',
		'image_folder':'img/',
		'feat':sys.argv[1],		# SURF or HIST
		'color':sys.argv[2],	# I or r or g
		'k':32}

# meta data
names = ['file_name','description','tax1','tax2','tax3']
meta = pd.read_csv(os.path.join(opts['base_folder'],'meta/clothes.csv'), header=0, delimiter=',', dtype=object, names=names)
filepaths_ = [os.path.join(opts['base_folder'],opts['image_folder'],f)+'.jpg' for f in meta.file_name]
filepaths = [f for f in filepaths_ if os.path.exists(f)]

# build index
if do_build:
	indexer = ip.VisualObjectMatcher(dict({'max_num_db_images':6000}.items()+opts.items()), False)
	indexer.build_index(filepaths)

# match example: best match should have a distance of 0
matcher = ip.VisualObjectMatcher(opts)
for i in range(0,10):
	print matcher.match(filepaths[i])