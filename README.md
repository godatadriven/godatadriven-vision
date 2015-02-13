# Visual search with python+OpenCV

Rank your product catalog based on visual similarity to a query photo.

godatadriven-vision
|-app	(webapp stuff)
|-lib	(core image processing and machine learning functionality plus associated utilities)
|-proc	(example scripts using the stuff in lib/)

# Setup

TODO

# Method

- Several options are considered:
	- dense / keypoint sampling (in the intensity image)
	- photometric representation: color / intensity (where color is normalized red and green)
		- the method should be repeated per image channel, for now
	- texture / pixel values
- Ranking proceeds by combining the ranks of every option combo