import time
import queue
import threading
import numpy as np
import configparser
from .Crypt import *
from .Sqlite3 import *
from .FaceMatcher import *
from .FaceDetector import *
from .EmotionDetector import *

def getDistance(p1,p2):
    p1=np.array(p1)
    p2=np.array(p2)
    distance=np.linalg.norm(p1-p2)
    return distance

class FaceManager:

    def __init__(self,radiusRate=2,configPath=r'config.ini',databasePath=r'DataBase.db3'):
        self.configPath=configPath
        self.databasePath=databasePath
        self.radiusRate=radiusRate
        self.start()
        
    def detect(self,frame):
        self.frameQueue.put(frame)
        while self.faceInfosQueue.empty()==False and self.running:
            self.lastestFaceInfos=self.faceInfosQueue.get()
        return self.lastestFaceInfos
        
    def loadData(self):
        datas=self.db.executeFetchall('SELECT faceEndcoding,username FROM user')
        cryptFaceEncodings=[data[0] for data in datas]
        names=[data[1] for data in datas]
        key = bytes.fromhex(self.config.get('Crypt', 'key'))
        iv = bytes.fromhex(self.config.get('Crypt', 'iv'))
        aas=ArrayAES256(key,iv,128,np.float64)
        faceEncodings=[aas.decrypt(ciphertext) for ciphertext in cryptFaceEncodings]
        self.fm.loadFaceEncodings(faceEncodings,names)
        
    def start(self):
        self.config = configparser.ConfigParser()
        self.config.read(self.configPath)
        self.db=DataBase(self.databasePath)
        self.fm = FaceMatcher()
        self.fd = FaceDetector()
        self.ed = EmotionDetector()
        self.loadData()
        self.faceInfos=[]
        self.processQueues = []
        self.finishedQueues = []
        self.lastestFaceInfos=[]
        self.frameQueue = queue.Queue()
        self.faceInfosQueue = queue.Queue()
        self.running=True
        tdfi=threading.Thread(target=self.detectFaceInfos,daemon = True)
        self.threads=[tdfi]
        self.tm=threading.Thread(target=self.threadManager,daemon = True)
        self.tm.start()
        
    def close(self):
        self.running=False
        self.tm.join()
        for thread in self.threads:
            if thread.is_alive():
                thread.join()
        #self.db.close()
    
    def reset(self):
        self.close()
        self.start()
        
    def threadManager(self):
        while self.running:
            for thread in self.threads:
                if thread.is_alive()==False:
                    thread.daemon = True
                    thread.start()
            time.sleep(1)
            
    def detectFaceInfos(self):
        while self.running:
            frame=None
            while self.frameQueue.empty()==False and self.running:
                frame=self.frameQueue.get()
            if frame is not None:
                faces, boxes = self.fd.detect(frame)
                for i in range(len(faces)):
                    self.updateFaceInfo(faces[i],boxes[i])
                self.faceInfosQueue.put(self.faceInfos)
        
    def updateFaceInfo(self,face,box):
        x,y,w,h=box
        for i in range(len(self.finishedQueues)):
            while self.finishedQueues[i].empty()==False and self.running:
                index,value=self.finishedQueues[i].get()
                self.faceInfos[i][index]=value
        for i in range(len(self.faceInfos)):
            box2,name,emotion=self.faceInfos[i]
            x2,y2,w2,h2=box2
            distance=getDistance([x+w/2,y+h/2],[x2+w2/2,y2+h2/2])
            if distance<np.max([w2,h2])/2*self.radiusRate:
                self.faceInfos[i][0]=box
                self.processQueues[i].put(face)
                return
        self.faceInfos.append([box,None,None])
        self.processQueues.append(queue.Queue())
        self.finishedQueues.append(queue.Queue())
        self.processQueues[len(self.faceInfos)-1].put(face)
        thread = threading.Thread(target=self.detectFaceInfo,args=(len(self.faceInfos)-1,self.processQueues[-1],self.finishedQueues[-1]),daemon = True)
        self.threads.append(thread)
        
    def detectFaceInfo(self,index,processQueue,finishedQueue):
        name=None
        while self.running:
            face=None
            if processQueue.empty()==False:
                face=processQueue.get()
                if self.faceInfos[index][1] is None and name is None:
                    name=self.fm.getMutiMatchFaceLabel(face)
                    if name is not None:
                        finishedQueue.put([1,name])
            if face is not None:
                emotion = self.ed.detect(face)
                finishedQueue.put([2,emotion])
            time.sleep(1)