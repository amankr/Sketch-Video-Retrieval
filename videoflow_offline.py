from decaf.scripts.imagenet import DecafNet
import numpy,scipy,PIL,csv,glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import os
import cv2,scipy
import pickle
from mlabwrap import mlab

ucfloc = 'decaf/KitchenData/'
imgnetPath = 'decaf/imagenet_pretrained/'
flowdir = 'flowdata/'

net = DecafNet(imgnetPath+'imagenet.decafnet.epoch90', imgnetPath+'imagenet.decafnet.meta')
pca = PCA(n_components=20)

class Feature():
	def __init__(self, decaf,category,_id):
		self.decaf = decaf
		self.category = category
		self.path = _id

def imToNumpy(img):
	return numpy.asarray(PIL.Image.open(img))

def getFeature(img):
	scores = net.classify(img, center_only=True)
	feature = net.feature('fc6_cudanet_out')
	return feature[0]
'''
def getFlow(f):
	flow = None
	flowsum = None
	cap = cv2.VideoCapture(f)
	ret, frame1 = cap.read()
	if frame1 == None:
		return flow
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	while(1):
		ret, frame2 = cap.read()
		if frame2==None:
			break
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		newfl = np.sqrt(np.add(flow[:,:,0]*flow[:,:,0],flow[:,:,1]*flow[:,:,1]))
		mn = np.amin(newfl)
		mx = np.amax(newfl)
		thresh = mx - (15*(mx-mn)/100)
		ids = newfl < thresh
		newfl[ids] = 0

		if flowsum == None:
			flowsum = newfl
		else:
			flowsum = np.add(flowsum,newfl)	
		prvs = next

	flowsum = (255-flowsum)
	_name = (f.split('/')[-1]).split('.')[0] + '.jpg'
	_path = flowdir+_name
	scipy.misc.imsave(_path, flowsum)
	cap.release()
	return _path
'''
def getFlow(f,name):
	flow = None
	flowsum = None
	prvs = None
	c = 0
	print len(f)
	for fr in f:
		x = fr #for kitchen dataset
		#x = cv2.imread(fr,0)
		print c
		c += 1
		if prvs == None:
			prvs = x
			continue
		next = x
		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		newfl = np.sqrt(np.add(flow[:,:,0]*flow[:,:,0],flow[:,:,1]*flow[:,:,1]))
		mn = np.amin(newfl)
		mx = np.amax(newfl)
		thresh = mx - (15*(mx-mn)/100)
		ids = newfl < thresh
		newfl[ids] = 0

		if flowsum == None:
			flowsum = newfl
		else:
			flowsum = np.add(flowsum,newfl)	
		prvs = next

	flowsum = (255-flowsum)
	#_name = name+((f[0]).split('/')[-1]).split('.')[0] + '.jpg'
	_path = flowdir+ name.split('.')[0]+'.jpg'
	scipy.misc.imsave(_path, flowsum)
	return _path

def getFrames(cap):
	f = []
	tot = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	for _ in range(tot):
		ret,frame = cap.read()
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		f.append(frame)
	return f

def createFlowDb(source):
	db = []
	category = 0
	for sports in os.listdir(source):
		path = os.path.join(source,sports)
		for videoDir in os.listdir(path):
			path2 = os.path.join(path,videoDir)
			print path2
			'''
			files = glob.glob(os.path.join(path2,'*.jpg'))
			files.sort()
			'''
			cap = cv2.VideoCapture(path2)
			files = getFrames(cap)
			_path = getFlow(files,videoDir)
			del files
			if _path == None:
				continue
			_flow = cv2.imread(_path,0)
			img = cv2.medianBlur(_flow,5)
			th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
			db.append(Feature(getFeature(th3),category,sports))
			
			cap.release()

		category = category+1
	return db

def listtoFile(lst,fname):
	with open(fname,'wb') as m:
		writer = csv.writer(m)
		writer.writerows(lst)

def DbToFile(db):
	ft = []
	cat = []
	names = []
	[ft.append(x.decaf) for x in db]
	[cat.append([x.category]) for x in db]
	for x in db:
		_t = []
		_t.append(x.category)
		_t.append(x.path)
		names.append(_t)

	ft = numpy.asarray(ft)
	pca.fit(ft)
	ft = pca.transform(ft)
	listtoFile(ft,'Flowfeature.csv')
	listtoFile(cat,'Flowclass.csv')
	listtoFile(names,'Flowinverse.csv')
	print "flow features written to files"

def dumpPCA(pc):
	with open('pcaFlowData.pkl', 'wb') as output:
		pickle.dump(pc, output, pickle.HIGHEST_PROTOCOL)

def main():
	myDb = createFlowDb(ucfloc)
	DbToFile(myDb)
	dumpPCA(pca)
	print mlab.LmnnFlowSave()

if __name__ == '__main__':
	main()
