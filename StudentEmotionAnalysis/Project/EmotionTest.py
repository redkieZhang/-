import os
from FaceManager import *
from VideoProcessing import *
import time

def resize(img,rate):
    rimg=cv2.resize(img,(int(img.shape[1]*rate),int(img.shape[0]*rate)))
    return rimg

fm = FaceManager()
vp = VideoProcessing(r'C:\Users\Alan\Desktop\3.mp4')
cv2.namedWindow('window_frame')
while True:
    
    frame = vp.read()
    start2=time.time()
    #frame=resize(frame,0.5)
    
    #faces, boxes = fd.detect(frame)
    
    faceInfos=fm.detect(frame)
    if time.time()-start2>0.01:
        print(time.time()-start2)
    for i in range(len(faceInfos)):
        box,name,emotion=faceInfos[i]
        x,y,w,h=box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)
        #start=time.time()
        #faceLabel,emotion=fb.getFaceName(faces[i],boxes[i])
        #if time.time()-start>0.01:
        #    print('bad',time.time()-start)
        # 在矩形框上部，输出分类文字
        if name is None:
            name='None'
        if emotion is None:
            emotion='None'
        cv2.putText(frame,name+' : '+emotion,(x+w//2,y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
    try:
        # 将图片从内存中显示到屏幕上
        cv2.imshow('window_frame', frame[:,:,[2,1,0]])
    except:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('e'):
        print('正在重置')
        fm.reset()
        print('重置成功')

vp.close()
cv2.destroyAllWindows()
