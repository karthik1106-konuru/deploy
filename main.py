
from flask import Flask,jsonify,request
import librosa
import numpy as np
from keras.models import load_model
from yolo_detection_images import detectWifiRouter
from numpy import asarray
from PIL import Image, ImageOps

app=Flask(__name__)
ans=[]
classes=["Cpu_Fan_Noise","Ram_Disconnected_Sound"]
model_obj=load_model("my_model.h5")

# @app.route('/')
@app.route('/',methods=['POST','GET'])
def prediction():
    if request.method == 'GET':
        if len(ans)!=0:
            return str(ans[-1])
        else:
            return "send your audio format"
    if request.method=='POST':
        if request.files:
            if 'file' in request.files:
                f=request.files.get('file')
                x, sr = librosa.load(f, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
                mfccs = mfccs.reshape(1, -1)
                # print(mfccs)
                predict_prob = model_obj.predict(mfccs)
                # print(predict_prob)
                predicted_label = np.argmax(predict_prob, axis=1)
                # print(predicted_label)
                for i in predicted_label:
                    # print(classes[i])
                    ans.append(classes[i])
                return str(ans[-1])
            if 'image' in request.files:
                image = Image.open(request.files['image'])
                numpydata = asarray(image)
                result = detectWifiRouter(numpydata)
                return result
        else:
            return "Send Your File"
        return "files not save"

if __name__=="__main__":
    app.debug=True
    app.run()
