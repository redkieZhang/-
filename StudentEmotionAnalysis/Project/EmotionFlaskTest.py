import os
from FaceMatcher import *
from FaceDetector import *
from VideoProcessing import *
from EmotionDetector import *
from flask import Flask, render_template, Response
from flask_httpauth import HTTPBasicAuth
from VideoProcessing import *

app = Flask(__name__)

def resize(img,rate):
    rimg=cv2.resize(img,(int(img.shape[1]*rate),int(img.shape[0]*rate)))
    return rimg

fm = FaceMatcher()
fd = FaceDetector()
vp = VideoProcessing(r'C:\Users\Alan\Desktop\3.mp4')
ed = EmotionDetector()
#cv2.namedWindow('window_frame')
faceDir=r'C:\Users\Alan\Desktop\model\face'
faces=[cv2.imread(os.path.join(faceDir,file)) for file in os.listdir(faceDir)]
faceLabels=['Sam','Lucy','Alan','Charlies','Hilary','Kitty','Ben']
fm.loadFaces(faces,faceLabels)
#trackers = cv2.legacy.MultiTracker_create()
def generate_frames():
    while True:
        frame = vp.read()
        #frame=resize(frame,0.5)
        faces, boxes = fd.detect(frame)
        for i in range(len(faces)):
            x,y,w,h=boxes[i]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(84,255,159),2)
            faceLabel=fm.getMutiMatchFaceLabel(faces[i])
            emotion_mode = ed.detect(faces[i])
            # 在矩形框上部，输出分类文字
            if faceLabel is None:
                faceLabel='None'
                #cv2.putText(frame,faceLabel,(x+w//2,y+h//2-20), cv2.FONT_HERSHEY_SIMPLEX, .7,(0,0,255),1,cv2.LINE_AA)
            cv2.putText(frame,faceLabel+' : '+emotion_mode,(x+w//2,y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame[:,:,[2,1,0]])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    #try:
    #    #cv2.resizeWindow('window_frame', 800, 600)
    #    # 将图片从内存中显示到屏幕上
    #    cv2.imshow('window_frame', frame[:,:,[2,1,0]])
    #except:
    #    continue
        
    # 按q退出
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

#vp.close()
#cv2.destroyAllWindows()

# 主页路由
@app.route('/')
#@auth.login_required
def index():
    return render_template('index.html')

# 视频流路由
@app.route('/video_feed')
#@auth.login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
