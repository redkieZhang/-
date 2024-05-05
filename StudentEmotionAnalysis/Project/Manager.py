from .FaceManager import *
from .VideoProcessing import *

class Manager:

    def __init__(self,videoPath):
        self.videoPath=videoPath
        self.start()
        
    def detect(self):
        frame = self.vp.read()
        pieImg = None
        emotions=[]
        usernames=[]
        if frame is not None:
            faceInfos=self.fm.detect(frame)
            for i in range(len(faceInfos)):
                box,name,emotion=faceInfos[i]
                x,y,w,h=box
                cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)
                if name is None:
                    name='None'
                if emotion is None:
                    emotion='None'
                else:
                    emotions.append(emotion)
                    usernames.append(name)
                text_size, _ = cv2.getTextSize(name+' '+emotion, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                text_width, text_height = text_size
                y+=int(h/4)
                cv2.rectangle(frame,(x+w//2-int(text_width//1.8),y+h-int(text_height*0.5)),(x+w//2+int(text_width//1.8),y+h+int(text_height*1.5)),(0,0,0),-1)
                cv2.putText(frame,name+' '+emotion,(x+w//2-text_width//2,y+h+text_height), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
        return frame,emotions,usernames
        
    def close(self):
        self.fm.close()
        self.vp.close()
        
    def start(self):
        self.fm = FaceManager()
        self.vp = VideoProcessing(self.videoPath)
        
    def reset(self):
        self.close()
        self.start()