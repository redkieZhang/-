import cv2
import numpy as np
import face_recognition

class FaceMatcher:

    def __init__(self,threshold=0.5):
        self.faceEncodings=None
        self.labels=None
        self.threshold=threshold

    def match(self,face1,face2):
        faceEncoding1=self.getFaceEncoding(face1)
        faceEncoding2=self.getFaceEncoding(face2)
        distance=self.matchFaceEncoding(faceEncoding1,faceEncoding2)
        if distance<self.threshold:
            return True
        else:
            return False
        
    def getMutiMatchFaceLabel(self,face):
        faceEncoding=self.getFaceEncoding(face)
        label=self.getMutiMatchFaceEncodingLabel(faceEncoding)
        return label
        
    def getMutiMatchFaceEncodingLabel(self,faceEncoding):
        index=self.getMutiMatchIndex(faceEncoding,self.faceEncodings)
        if index is not None:
            return self.labels[index]
        else:
            return None
        
    def loadFaces(self,faces,labels):
        self.faceEncodings=self.getFaceEncodings(faces)
        self.labels=labels
        
    def loadFaceEncodings(self,faceEncodings,labels):
        self.faceEncodings=faceEncodings
        self.labels=labels
        
    def getMutiMatchIndex(self,faceEncoding,faceEncodings):
        if faceEncoding is None or faceEncodings is None:
            return None
        minDistance=None
        minIndex=None
        for i in range(len(faceEncodings)):
            distance=self.matchFaceEncoding(faceEncoding,faceEncodings[i])
            if distance is None or distance>self.threshold:
                continue
            if minDistance is None or distance<minDistance:
                minDistance=distance
                minIndex=i
        return minIndex
        
    def getFaceEncodings(self,faces):
        faceEncodings = []
        for face in faces:
            faceEncoding = self.getFaceEncoding(face)
            faceEncodings.append(faceEncoding)
        return faceEncodings
        
    def getFaceEncoding(self,face):
        try:
            faceEncoding = face_recognition.face_encodings(face,model="large")[0]
            return faceEncoding
        except:
            return None
        
    def matchFaceEncoding(self,faceEncoding1,faceEncoding2):
        if faceEncoding1 is None or faceEncoding2 is None:
            return None
        distance = np.linalg.norm(faceEncoding1 - faceEncoding2)
        return distance
    