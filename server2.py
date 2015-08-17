'''
    Simple socket server using threads
'''
import socket
import sys

from mlabwrap import mlab
import pickle
from decaf.scripts.imagenet import DecafNet
import numpy,scipy,PIL,csv
from PIL import Image
from sklearn.decomposition import PCA

#import thread
from thread import *
HOST = ''   # Symbolic name meaning all available interfaces
PORT = 8080 # Arbitrary non-privileged port
cnt = 0
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()
         
print('Socket bind complete')
#Start listening on socket
s.listen(10)
print('Socket now listening')

#################################

ucfloc = 'decaf/ucfaction/'
imgnetPath = 'decaf/imagenet_pretrained/'
numFrames = 2

#load Imagenet training dataset
net = DecafNet(imgnetPath+'imagenet.decafnet.epoch90', imgnetPath+'imagenet.decafnet.meta')

def loadPCA():
    with open('pcaData.pkl', 'rb') as input:
        pca = pickle.load(input)
    return pca

def loadFlowPCA():
    with open('pcaFlowData.pkl', 'rb') as input:
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

def checkImage(querryImage):
    pca = loadPCA()
    sketch = pca.transform(getFeature(imToNumpy(querryImage)))
    listtoFile(sketch,'querry_tmp.csv')
    result = mlab.LmnnQuerry()
    dic = {}
    l1 =[]
    l2 =[]
    for x in result.vid:
        l1.append(int(x[0]))
    for y in result.dist:
        l2.append(y[0])
    mx = max(l2)
    for x in range(len(l2)):
        l2[x] = mx - l2[x]

    # TODO : video query - 300 , flow qwery - 100

    for i in range(len(result.vid)):
        dic[i] = 0
    for i in range(len(result.vid)):
        dic[l1[i]] = l2[i]

    dic2 = {}
    c = 0
    while c<100:
        dic2[c] = (dic[3*c]+dic[3*c+1]+dic[3*c+2])/3
        c += 1
    return dic2
    #for x in range(len(result.vid)):
    #    dic[int(result.vid[x][0])] = result.dist[x][0]
    #print int(result[0][0])
    #return int(result[0][0])

def checkFlow(querryImage):
    pca = loadFlowPCA()
    sketch = pca.transform(getFeature(imToNumpy(querryImage)))
    listtoFile(sketch,'fquerry_tmp.csv')
    result = mlab.LmnnFlowQuerry()
    #result = mlab.LmnnQuerry()
    dic = {}
    l1 =[]
    l2 =[]
    for x in result.vid:
        l1.append(int(x[0]))
    for y in result.dist:
        l2.append(y[0])
    mx = max(l2)
    for x in range(len(l2)):
        l2[x] = mx - l2[x]
    for i in range(len(result.vid)):
        dic[i] = 0
    for i in range(len(result.vid)):
        dic[l1[i]] = l2[i]
    return dic
    #print int(result[0][0])
    #return int(result[0][0])

def getScore(im1,im2):
    alpha = 0.3
    top = 5
    d1 = checkImage(im1)
    d2 = checkFlow(im2)
    score = []

    print len(d1),len(d2)

    for x in d1:
        print x,d1[x],d2[x]
        s = (alpha * d1[x] + (1 - alpha) * d2[x] , x)
        score.append(s)
    score.sort()
    
    videos = ""
    for x in range(top):
        videos += str(score[x][1])+":"

    return videos


#################################


result = 0
mylock =  False
#Function for handling connections. This will be used to create threads
def clientthread(conn,cnt):
    global result,mylock
    mylock = False
    #infinite loop so that function do not terminate and thread do not end.
    fname = "img" + str(cnt) + ".jpg"
    f = open(fname,'wb')
    while True:
         
        #Receiving from client
        data = conn.recv(1024)
        f.write(data)
        if not data:
            break
     
    f.close();
    #result = checkImage(fname)
    mylock = True
    #came out of loop
    conn.close()

#Function for handling connections. This will be used to create threads
def clientthread2(conn,cnt):
   # global result,mylock
    print "clientthread2"
    fname = "img" + str(cnt) + ".jpg"
    fname1 = "img" + str(cnt+1) + ".jpg"
    #result = checkImage(fname)
    #fresult = checkFlow(fname1)
    result = getScore(fname,fname1)

    data = conn.recv(1024)
    print data,result
    if data == "send val":
        print result
        conn.send(result+"\n")#str(result)+" "+str(fresult)+"\n")
        #conn.send("25:28:15:80:89\n")
        print "result sent"
#now keep talking with the client
while 1:
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    cnt += 1
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    if cnt%3 == 0 :
        start_new_thread(clientthread2 ,(conn,cnt-2))    
    else:
        #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
        start_new_thread(clientthread ,(conn,cnt,))
 
s.close()
