from decaf.scripts.imagenet import DecafNet
import numpy,scipy,PIL,csv,glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import os
import cv2
import pickle
from mlabwrap import mlab
import shutil,sys

#set paths
ucfloc = 'decaf/KitchenData/'
imgnetPath = 'decaf/imagenet_pretrained/'
videoout = 'copyToDevice/'
numFrames = 3

#load Imagenet training dataset
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

def getFrames(cap):
	f = []
	tot = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,tot/2)
	for _ in range(numFrames):
		ret,frame = cap.read()
		f.append(frame)
	return f
	'''
	for _ in range(tot):
		ret,frame = cap.read()
		f.append(frame)
	ans = []
	ans.append(f[tot/2])
	ans.append(f[tot/2 + 1])
	ans.append(f[tot/2 + 2 ])
	del f
	return ans
	'''

def createDB(source):
	db = []
	category = 0
	count = 0
	for sports in os.listdir(source):
		path = os.path.join(source,sports) # folder path
		for videoDir in os.listdir(path):
			path2 = os.path.join(path,videoDir)
			'''
			files = os.listdir(path2)
			for f in files:
				if f.endswith('.avi'):
					print path2,f
					cap = cv2.VideoCapture(os.path.join(path2,f))
					frames = getFrames(cap)
					for fr in frames:
						if fr is not None:
							print fr
							#img = cv2.imdecode(fr,cv2.COLOR_BGR2GRAY)
							img = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
							print img
							img = cv2.medianBlur(fr,5)
							ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
							th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
							db.append(Feature(getFeature(th3),category,sports))#os.path.join(path2,f)))
			'''
			f = videoDir
			
			if f.endswith('.avi'):
				print f,path2
				cap = cv2.VideoCapture(path2)
				frames = getFrames(cap)
				for fr in frames:
					if fr is not None:
						#print fr
						#img = cv2.imdecode(fr,cv2.COLOR_BGR2GRAY)
						img = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
						img = cv2.convertScaleAbs(img)
						print cv2.imwrite('t1.jpg',img)
						img2 = cv2.imread('t1.jpg',0)
						img2 = cv2.medianBlur(img2,5)
						#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
						th3 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
						db.append(Feature(getFeature(th3),category,sports))#os.path.join(path2,f)))
						cv2.imwrite('dataimg/'+f.split('.')[0]+'.jpg',th3)
						del img
						del img2
						del th3

				#shutil.copy2(path2, videoout+str(count)+'.avi')
				
				cap.release()
				del frames
			
			print sys.getsizeof(db)

			'''
			fl = glob.glob(os.path.join(path2,'*.avi'))
			for x in fl:
				shutil.copy2(x, videoout+str(count)+'.avi')

			files = glob.glob(os.path.join(path2,'*.jpg'))
			files.sort()
			fcount = 0
			for f in files:
				if fcount>numFrames:
					break
				img = cv2.imread(f,0)
				img = cv2.medianBlur(img,5)
				th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
				db.append(Feature(getFeature(th3),category,sports))#os.path.join(path2,f)))
				#cv2.imwrite('dataimg/'+f.split('/')[-1],th3)
				fcount = fcount+1
			'''

			count += 1
		
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
	#[names.append([x.path]) for x in db]
	for x in db:
		_t = []
		_t.append(x.category)
		_t.append(x.path)
		names.append(_t)

	ft = numpy.asarray(ft)
	pca.fit(ft)
	ft = pca.transform(ft)
	listtoFile(ft,'feature.csv')
	listtoFile(cat,'class.csv')
	listtoFile(names,'inverse.csv')
	print "features written to files"

def dumpPCA(pc):
	with open('pcaData.pkl', 'wb') as output:
		pickle.dump(pc, output, pickle.HIGHEST_PROTOCOL)

def main():
	myDb = createDB(ucfloc)
	DbToFile(myDb)
	dumpPCA(pca)
	print mlab.LmnnSave()

if __name__ == '__main__':
	main()