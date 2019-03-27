from PIL import Image
import numpy as np
import flask
import io
import pickle
import cv2
import keras.applications.imagenet_utils
import subprocess
from subprocess import Popen, PIPE

app = flask.Flask(__name__)

f = open("hindi_final_lookup.txt", 'r')
answer = {}
count = 1
for line in f:
    k= int(line.strip())
    answer[count] = k
    count+=1

f.close()


@app.route("/predict", methods=["POST"])
def predict():
        
        data = {"success": False}

        if flask.request.method == "POST":
                if flask.request.files.get("image"):
                        # read the image in PIL format
                        image = flask.request.files["image"].read()
                        image = Image.open(io.BytesIO(image))
                        image = np.array(image)

                        #pdb.set_trace()
                        if image.ndim==3:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite("temp.jpg", image)
                        
                        session = subprocess.Popen(['th', 'test_standalone.lua', 'temp.jpg'], stdout=PIPE, stderr=PIPE)
                        stdout, stderr = session.communicate()
  
                        if stderr:
                            data["results"] = "CODE DIDNT WORK"
                        else:
                            data["success"] = True
                            predicted =  stdout[stdout.find("Recognized text: "):]
                            predicted = stdout[stdout.find(":")+2:]
                            predicted = predicted.strip()
                            string = ""
                            for ch in predicted.split():
                                if ch!=0 and ch!='0':
                                    string+=  unichr(answer[int(ch)])
                            string.encode('utf-8')
                            data["results"] = string.encode('utf-8')


        # return the data dictionary as a JSON response
        return flask.jsonify(data)

#if __name__ == "__main__":
        #Need to check if we are running on a server
        
app.run()
