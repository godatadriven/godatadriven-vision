# By Ivo Everts

import cv2
import numpy as np
import os
import json
import pandas as pd
import pickle
import util
import csv
import pdb

class VisualObjectMatcher(object):
	"""
	This class can be used to create and/or invoke models for visual object matching.
	Experiments and production mixed up!
	"""

	#####################################################################################
	# GENERAL SETTINGS																	#
	#####################################################################################

	# grayscale or color images
	COLOR_MODE 				= cv2.IMREAD_GRAYSCALE # CV_LOAD_IMAGE_COLOR
	
	# file paths + naming
	MODEL_FOLDER 			= 'model/'
	JSON_OUTPUT_FOLDER 		= 'rank-output/'


	#####################################################################################
	# CLASS MEMBERS + DEFAULT VALUES													#
	#####################################################################################
	# NOTE: this is not pythonic
	class_members = {'color'					: 'I',			# photometric representation, in {'I','r','g'}
					 'sampler'					: 'KP', 		# feature detection in {'KP','DE'} ({'KEYPOINT','DENSE'})
					 'feat'						: 'SURF', 		# feature detection+extration, in {'ORB','SURF','SIFT','HIST'}
					 'model'					: 'VLAD',		# image representation, in {'VLAD','BOW'}
					 'k'						: 16,			# size of visual vocabulary, ORB+VLAD->4, SURF+VLAD->32, BOW->1000s
					 'num_matches'				: 5,			# number of results to return per query image
					 'base_folder'				: '/',			# base folder of the project on the file system
					 'image_folder'				: '/',			# image folder relative to base folder
					 'image_extension'			: 'jpg',		# image extension (used in experiments)
					 'num_feats_for_codebook'	: 500000,		# number of features per cluster needed for creating a codebook
					 'max_num_db_images'		: 1000,			# limit the number of db images used in an experiment
					 'save_model'				: True}			# save models to model folder


	#####################################################################################
	# CLASS METHODS																		#
	#####################################################################################
	
	def __init__(self, arg_dict, load_models=True):
		
		# check if a base folder was given
		if not arg_dict.has_key('base_folder'):
			print 'Warning: no base_folder given, using', self.class_members['base_folder']

		# initialize class members
		self.init_members(arg_dict)

		# sign of life
		print 'VisualObjectMatcher', self.signature

		# switch between experiment and production mode
		if load_models:
			self.codebook, self.traindata, self.trainfiles = self.load_models()
	

	# init class members from args or corresponding default values
	def init_members(self, args):
		# set class members
		for class_member in VisualObjectMatcher.class_members:
			if args.has_key(class_member):
				# use given setting
				value = args[class_member]
			else:
				# init to default value (static class var)
				value = VisualObjectMatcher.class_members[class_member]
			setattr(self, class_member, value)
		
		# strings of parameter settings
		self.codebook_string = self.feat + str(self.k)
		self.signature = self.codebook_string + self.model
		self.param_string = '_D' + str(self.max_num_db_images) + \
							'_M' + str(self.num_matches) + \
							'_'  + self.signature

		# set feature extractor
		self.featd = cv2.FeatureDetector_create(self.feat)
		# hack: always use harris
		# todo: use multiple feature detectors
		self.harris = cv2.FeatureDetector_create('HARRIS')
		self.featx = cv2.DescriptorExtractor_create(self.feat)

	# api call for matching one file
	# it's a quicky ...
	def match(self, query_filepath):
		
		query_descriptors, failures = self.describe_image_files([query_filepath], self.codebook)
		if sum(failures) > 0:
			print 'No image representation created for', query_filepath
			return None

		matches, distances = self.flann_match(query_descriptors, self.traindata, self.num_matches)
		return self.export_results(matches, distances, [query_filepath], self.trainfiles, False)

	def load_models(self):
		codebook = np.load(self.base_folder + self.MODEL_FOLDER + self.codebook_string + '.npy')
		traindata = np.load(self.base_folder + self.MODEL_FOLDER + self.signature + '.npy')
		trainfiles = pickle.load(open(self.base_folder + self.MODEL_FOLDER + self.signature,'r'))
		return (codebook, traindata, trainfiles)

	def save_models(self, codebook, db_descriptors, db_filepaths):
		util.mkdir(self.base_folder + self.MODEL_FOLDER)
		np.save(self.base_folder + self.MODEL_FOLDER + self.codebook_string, codebook)
		np.save(self.base_folder + self.MODEL_FOLDER + self.signature, db_descriptors)
		pickle.dump(db_filepaths, open(self.base_folder + self.MODEL_FOLDER + self.signature,'w'))

	# run an experiment
	def build_index(self, db_filepaths=None):

		if db_filepaths is None:
			# list train files
			db_filepaths = util.list_files(os.path.join(self.base_folder, self.image_folder), self.image_extension)
		db_filepaths = db_filepaths[:min(len(db_filepaths),self.max_num_db_images)]
		
		# load codebook, if applicable
		cb_file = self.base_folder + self.MODEL_FOLDER + self.codebook_string + '.npy'
		codebook = None
		if os.path.isfile(cb_file):
			codebook = np.load(cb_file)
			
		# build model
		codebook, db_descriptors, failures = self.build_model(db_filepaths, codebook)
		db_descriptors = db_descriptors[~failures]
		db_filepaths = util.stridx(db_filepaths, ~failures)
		
		# save whole model as needed in a live scenario
		if self.save_model:
			self.save_models(codebook, db_descriptors, db_filepaths)

	# detect and extract features in the given image
	# kp: detected keypoints
	# desc: descriptors, ensured to be of type float32 for distance computation
	# hkp = []#self.harris.detect(image)
	def extract_features_from_image(self, image):
		# sample locations
		if self.sampler == "KP":
			# keypoint detector
			kp = self.featd.detect(image.intensity_image)
		else:
			# dense sampling
			kp = self.grid_samples(image.intensity_image.shape)
		# hack: set angle to 0
		for kp_ in kp:
			kp_.angle = 0
		# consider different descriptor types
		if self.feat == "HIST":
			f = self.histogram_descriptors_from_keypoints(kp, image.t_image)
		else:
			kp, f = self.featx.compute(image.intensity_image, kp)
		# convert to convenient format
		if f is not None:
			f.astype(np.float32, copy=False)
		return kp, f

	# wrapper for reading an image and extracting keypoint descriptors
	def extract_features_from_filename(self, filename, color):
		image = self.read_image(filename, color)
		if image is not None:
			kp, f = self.extract_features_from_image(image)
			if f is None:
				print 'No features detected in', filename
			return kp, f
		return None

	# build a visual vocabulary by clustering the features from the given files
	def extract_features_for_codebook(self, filepaths):
		# number of descriptors to extract per file
		num_feats_per_file = self.num_feats_for_codebook / len(filepaths)
		# actual predicted number of features
		num_feats_total = num_feats_per_file * len(filepaths)
		# do once for determining feature dimension (and do not care that this is performed twice:)
		kp, f = self.extract_features_from_filename(filepaths[0], self.color)
		# init data matrix
		data = np.empty([num_feats_total, f.shape[1]], dtype=float)
		# actual number of features
		num_feats = 0
		for filename in filepaths:
			# extract features
			kp, f = self.extract_features_from_filename(filename)
			if f is None or f.shape[0] == 0:
				continue
			# add to data matrix
			idx = util.random_sample(f.shape[0], num_feats_per_file)
			data[num_feats:num_feats+len(idx),:] = f[idx,:]
			num_feats += len(idx)
		# return appropriate data
		return np.float32(data[0:num_feats,:])

	# create a codebook by clustering a matrix of features
	def create_codebook(self, data):
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		compactness, labels, centers = cv2.kmeans(data, self.k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
		return centers

	# create codebook descriptor by projecting image features on codebook
	def describe_image(self, image, codebook):
		kp, f = self.extract_features_from_image(image)
		
		# return None if no features were found
		if f is None:
			return None
		
		cb_idx = self.get_closest_idx(f, codebook)
		image_descriptor = None
		# standard bag-of-words: project 
		if self.model == 'BOW':
			image_descriptor = np.histogram(cb_idx, range(0,self.k+1), density=True)[0]
		# store average element-wise distance to cluster centers
		elif self.model == 'VLAD':
			fdim = codebook.shape[1]
			image_descriptor = np.zeros(self.k*fdim, dtype=float)
			for k in range(0,self.k):
				cb_f = f[cb_idx == k,:]
				if cb_f.shape[0] > 0:
					vlad_component = codebook[k,:] - cb_f.mean(axis=0)
					image_descriptor[k*fdim:(k+1)*fdim] = vlad_component / np.linalg.norm(vlad_component)
		else:
			print 'Unknown model:', self.model
		return image_descriptor / np.linalg.norm(image_descriptor)

	# create codebook descriptors for all given images
	def describe_image_files(self, filepaths, codebook):
		dim = self.k
		if self.model == 'VLAD':
			dim = codebook.shape[0] * codebook.shape[1]
		descriptors = np.zeros((len(filepaths),dim), dtype=float)
		failures = np.zeros(descriptors.shape[0], dtype=bool)
		i = 0
		for filepath in filepaths:
			image = self.read_image(filepath, self.color)
			if image is not None:
				descriptor = self.describe_image(image, codebook)
				if descriptor is not None:
					descriptors[i,:] = descriptor
				else:
					failures[i] = True
			else:
				failures[i] = True
			i += 1
		return descriptors, failures

	# create traindata; create codebook if necessary
	def build_model(self, db_filepaths, codebook=None):
		if codebook is None:
			# create codebook
			util.tstart('extract features for codebook')
			features_for_codebook = self.extract_features_for_codebook(db_filepaths)
			util.tstop()
			util.tstart('create codebook')
			codebook = self.create_codebook(features_for_codebook)
			util.tstop()
		# apply codebook to database
		util.tstart('create database')
		db_descriptors, failures = self.describe_image_files(db_filepaths, codebook)
		util.tstop()
		# return
		return (codebook, db_descriptors, failures)

	# write matches to a json file
	def export_results(self, matches, distances, query_filepaths, db_filepaths, write_to_file=True):
	
		results = []
		for match_set, distance_set, query_filepath in zip(matches, distances, query_filepaths):
			result = dict()
			result['filename'] = query_filepath
			result['results'] = []
			for match, distance in zip(match_set, distance_set):
				db_filename = os.path.split(db_filepaths[match])[1]
				# harvest result data
				result['results'].append({'filename':db_filename,
										  'distance':float(distance)})
			results.append(result)
		
		# sort based on match_score
		json_result = json.dumps(results)
		if write_to_file:
			json_blob = open(self.base_folder + self.JSON_OUTPUT_FOLDER + self.param_string+'.json','w')
			json_blob.write(json_result)
			json_blob.close()
		return json_result

	def match_image_files(self, query_filepaths, codebook, db_descriptors, write_to_file=False, db_filepaths=None):
		util.tstart('match_image_files')
		query_descriptors, failures = self.describe_image_files(query_filepaths, codebook)
		# stop if no features were found
		if sum(failures) == len(query_filepaths):
			return (None, None, None)
		elif sum(failures) > 0:
			# subset
			query_descriptors = query_descriptors[~failures]
			query_filepaths = util.stridx(query_filepaths,~failures)
		# go
		util.tstop()
		util.tstart('flann matching')
		matches, distances = self.flann_match(query_descriptors, db_descriptors, self.num_matches)
		util.tstop()
		_ = self.export_results(matches, distances, query_filepaths, db_filepaths, write_to_file)
		return (matches, distances, query_filepaths)


	#####################################################################################
	# STATIC METHODS																	#
	#####################################################################################

	# read an image from disk
	@staticmethod
	def read_image(filename, color):
		print filename
		gdd_image = GDDImage(filename, color)
		if gdd_image.ok:
			return gdd_image
		return None

	# show an image and wait for key press
	@staticmethod
	def show_image(image, wname='image', wait=True):
		cv2.imshow(wname, image)
		if wait:
			cv2.waitKey()
			cv2.destroyAllWindows()

	# compute pairwise distances between the samples in each of the given matrices
	# this is a rewritten and vectorized version of the euclidean distance
	# note that this function assumes the dimensions to run along the rows
	@staticmethod
	def compute_distance_matrix(a, b):
		aa = np.sum(a**2, axis=0)
		bb = np.sum(b**2, axis=0)
		ab = np.dot(a.T, b)
		aa2 = np.repeat(aa.reshape([aa.size,1]), bb.size, axis=1)
		bb2 = np.repeat(bb.reshape([1,bb.size]), aa.size, axis=0)
		return np.sqrt(np.abs(aa2 + bb2 - 2*ab))

	# return the index in b of the row that is closest to each row in a 
	# this is a custom 1-NN classifier
	@staticmethod
	def get_closest_idx(a, b):
		return VisualObjectMatcher.compute_distance_matrix(a.T, b.T).argmin(axis=1)

	# find the n best matching rows in b for every row in a
	# this is still used because there seems to be a bug in opencv's flann_Index ... :()
	@staticmethod
	def flann_match(a, b, n):
		flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=1), dict(checks=5))
		matches = flann.knnMatch(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False), k=n)
		ms = np.zeros((a.shape[0],n), dtype=np.int32)
		ds = np.zeros((a.shape[0],n), dtype=np.float32)
		i = 0
		for match_set in matches:
			j = 0
			for match in match_set:
				ms[i,j] = match.trainIdx
				ds[i,j] = match.distance
				j += 1
			i += 1
		return (ms, ds)


class GDDImage(object):
	SCALE = 0.5 # in (0,1]
	COLOR2IDX = {'r':2,'g':1}
	def __init__(self, filepath, color, crop=False):
		setattr(self, 'color', color)
		self.bgr_image = cv2.imread(filepath, cv2.IMREAD_COLOR)
		if self.bgr_image is not None:
			ok = True
			if crop:
				# ignore white pixels, which are usually omnipresent in catalog images
				row, col, dim = np.where(np.logical_not(self.bgr_image==np.ones(self.bgr_image.shape,self.bgr_image.dtype)*255))
				row = np.unique(np.concatenate((row[0:len(row):3],row[1:len(row):3],row[2:len(row):3])))
				col = np.unique(np.concatenate((col[0:len(col):3],col[1:len(col):3],col[2:len(col):3])))
				self.bgr_image = self.bgr_image[min(row)+1:max(row)-1,min(col)+1:max(col)-1,:]
			# resize if applicable
			if self.SCALE != 1:
				self.bgr_image = cv2.resize(self.bgr_image, (0,0), fx=self.SCALE, fy=self.SCALE)
			# keep intensity image
			self.intensity_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2GRAY)
			# create color transformed image
			if color == 'I':
				# plain intensity
				self.t_image = self.intensity_image.astype(np.float32, copy=False) / 255.
			else:
				# normalized red or green
				self.t_image = self.bgr_image[:,:,self.COLOR2IDX[color]].astype(np.float32,copy=False) / np.sum(self.bgr_image,axis=2)
		else:
			ok = False
		setattr(self, 'ok', ok)