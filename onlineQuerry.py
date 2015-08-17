from mlabwrap import mlab
import pickle
from decaf.scripts.imagenet import DecafNet
import numpy,scipy,PIL,csv
from PIL import Image
from sklearn.decomposition import PCA


ucfloc = 'decaf/ucfaction/'
imgnetPath = 'decaf/imagenet_pretrained/'
numFrames = 2

#load Imagenet training dataset
net = DecafNet(imgnetPath+'imagenet.decafnet.epoch90', imgnetPath+'imagenet.decafnet.meta')

def loadPCA():
	with open('pcaData.pkl', 'rb') as input:
		pca = pickle.load(input)
	return pca

def imToNumpy(img):
	return numpy.asarray(PIL.Image.open(img))

def getFeature(img):
	scores = net.classify(img, center_only=True)
	feature = net.feature('fc6_cudanet_out')
	return feature

def listtoFile(lst,fname):
	with open(fname,'wb') as m:
		writer = csv.writer(m)
		writer.writerows(lst)

def main():
	querryImage = 'img5.jpg' # bhupkas tcp function
	pca = loadPCA()
	sketch = pca.transform(getFeature(imToNumpy(querryImage)))
	listtoFile(sketch,'querry_tmp.csv')
	result = mlab.LmnnQuerry()
	print int(result[0][0])

if __name__ == '__main__':
	main()