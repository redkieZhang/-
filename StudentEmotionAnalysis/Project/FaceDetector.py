import os
import cv2
curDir=os.path.dirname(os.path.abspath(__file__))

class FaceDetector:
    def __init__(self,model_path = os.path.join(curDir,'model/haarcascade_frontalface_default.xml')):
        self.face_detection = cv2.CascadeClassifier(model_path)
        
    def detect(self,frame):
        # 获得灰度图，并且在内存中创建一个图像对象
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 获取当前帧中的全部人脸
        boxes = self.face_detection.detectMultiScale(gray,1.3,5)
        faces=[]
        # 对于所有发现的人脸
        for (x, y, w, h) in boxes:
            # 获取人脸图像
            face = frame[y:y+h,x:x+w].copy()
            faces.append(face)
        return faces,boxes