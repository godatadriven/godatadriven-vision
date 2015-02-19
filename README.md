# Visual search with python+OpenCV


Rank your product catalog based on visual similarity to a query photo with our state-of- the-art-visualisation-starting-toolkit™. Just add water. 

<img src="/app/static/images/vision.png" alt="" width="100%">

## Contents

```
godatadriven-vision
|-app	flask webapp
|-lib	core image processing 
| 		machine learning functionality
| 		associated utilities
|-proc	example scripts using the stuff in lib/
```

## Installation 

### Mac Installation 

First make sure you follow all of the steps in [this link](http://www.jeffreythompson.org/blog/2013/08/22/update-installing-opencv-on-mac-mountain-lion/). This should ensure that opencv works. 

Then create a new virtual environment in the rootfolder of this git repo. I am using the old school way of virtualenv, but you could also use virualenvwrapper. 

```
virtualenv gdd_vision 
source gdd_vision/bin/activate 
```

Next we need to make sure we have all the required pip-packages. 

```
pip install -r requirements.txt
```

After this install has completed you should be able to confirm that the following code runs without errors:

```
(gdd_vision)$ python
Python 2.7.5 (default, Mar  9 2014, 22:15:05)
[GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.0.68)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv
>>> import cv2
>>> import numpy as np
```

You should also be able to do this in a python notebook. 

```
(gdd_vision)$ ipython notebook
```

### Linux Installation 

> This part of the readme might not be as complete. 

You need to follow these steps to get python2.7, pip, virtualenv and opencv to work on linux redhat. 

#### Install Python + Devtools 

```
yum install gcc gcc-c++.x86_64 compat-gcc-34-c++.x86_64 openssl-devel.x86_64 zlib*.x86_64
wget http://www.python.org/ftp/python/2.7/Python-2.7.tar.bz2
tar -xvjf Python-2.7.tar.bz2
cd Python*
./configure --prefix=/opt/python27
make
make install
```

Now make sure that the user can access python by appending the path. In ```.bash_profile``` make sure the following line is there. 

```
PATH=$PATH:$HOME/bin:/opt/python27/bin
```

Now make sure this file is sourced. 

```
source ~/.bash_profile
```

We now have a functional version of python 2.7. Now for the dependency management. 

```
curl -k https://bootstrap.pypa.io/get-pip.py | python2.7
```

Make sure that it works via. 

```
pip2.7 install requests
python2.7 
>>>> import requests
>>>> exit() 
pip2.7 install virtualenv
```

From here you should now be able to repeat similar steps done in the mac tutorial. 

## Setup

When everything has been properly installed you can run the app by telling flask to start running. From the `/app` directory: 

```
$ python app.py
```

Then point your browser to ```localhost:1234```.

# Method

- Several options are considered:
	- dense / keypoint sampling (in the intensity image)
	- photometric representation: color / intensity (where color is normalized red and green)
		- the method should be repeated per image channel, for now
	- texture / pixel values
- Ranking proceeds by combining the ranks of every option combo