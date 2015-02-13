# flask app
import os
import numpy
import time

# list certain file types in a folder
def list_files(folder, extension):
	if folder[-1] != '/':
		folder += '/'
	return [folder+filename for filename in os.listdir(folder) if filename.split('.').pop()==extension]

# sample a random subset from the given matrix shape
def random_sample(nrows, nsamples):
    if nsamples > nrows:
        return range(0,nrows)
    idx = numpy.arange(nrows)
    numpy.random.shuffle(idx)
    return idx[0:nsamples]

# extract article number from filename
def file2article(f):
	return os.path.split(f)[1].split('.')[0].split('_')[0]

# extract article numbers from filenames
def files2articles(fs):
	arts = list(fs)
	for i in range(0,len(fs)):
		arts[i] = file2article(fs[i])
	return arts

# timers
tstore = 0
def tstart(msg='...'):
    global tstore
    print 'Timing', msg
    tstore = time.time()
def tstop(msg=''):
    global tstore
    print round(time.time()-tstore,2),'sec',msg

# custom mkdir
def mkdir(f):
    if not os.path.exists(f):
        os.mkdir(f)

def stridx(string_array, logical_idx):
    return [s for s,i in zip(string_array, logical_idx) if i]