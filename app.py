from flask import Flask, render_template, request, session,Response, redirect
from isodate import parse_duration
import requests
import cv2
from keras.models import model_from_json
import numpy as np
import pickle
import webbrowser
from imutils.video import WebcamVideoStream

YOUTUBE_API_KEY= 'AIzaSyAZko-KXF6usQ5yTZHDWxfDrkxz6gyiwBg'

app = Flask(__name__)
webcam=cv2.VideoCapture(0)

emotions = {'happy': 0, 'sad': 0, 'fear': 0, 'neutral': 0, 'surprise': 0, 'angry': 0, 'disgust': 0}
emotion_list = []  # Your list of emotions of size 100
max_elements = 1000
max_emotion=''
global count

def generate_frames():   
    count=0
    json_file = open('emotiondetector.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.load_weights('emotiondetector.h5')
    haar_file = cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def extract_features(image):
        feature = np.array(image)
        feature = feature.reshape(1,48,48,1)
        return feature/255.0


    labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
    while count<=60:
        i,im = webcam.read()
        if not i:
            break
        else:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            try:
                for(p,q,r,s) in faces:
                    image = gray[q:q+s, p:p+r]
                    cv2.rectangle(im, (p,q), (p+r, q+s), (255,0,0), 2)
                    image=cv2.resize(image, (48,48))
                    img = extract_features(image)
                    pred = model.predict(img)
                    prediction_label = labels[pred.argmax()]
                    #print(prediction_label)
                    #cv2.putText(im, prediction_label)
                    cv2.putText(im,'% s' %(prediction_label), (p-10,q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
                    while len(emotion_list) < max_elements:
                        element = str(prediction_label)
                        if element.lower() == 'done':
                            break
                        emotion_list.append(element)
                
                ret,buffer = cv2.imencode('.jpg',im)
                frame = buffer.tobytes()
                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                #printing the list of emotions captured and printing the most recorded emotion
                #print("Final list:", emotion_list)
                # Iterate through the list and count each emotion
                for i in emotion_list:
                    if i in emotions:
                        emotions[i] += 1
                # Find the emotion with the maximum count
                max_emotion = max(emotions, key=emotions.get)
                print("The emotion repeated the most is:", max_emotion)
                count=count+1
    
            except cv2.error:
                print('Please check your webcam',cv2.error)
                break
    webcam.release()
    print("This is end of stream")
    open_youtube(max_emotion)
    

#========================================================================================================================          
def open_youtube(max_emotion):
    search_url = 'https://www.googleapis.com/youtube/v3/search'
    video_url = 'https://www.googleapis.com/youtube/v3/videos'
    videos = []

    search_params = {
            'key' : YOUTUBE_API_KEY,
            'q' : 'bollywood '+max_emotion+ ' songs',
            'part' : 'snippet',
            'maxResults' : 20,
            'type' : 'video'
        }

    r = requests.get(search_url, params=search_params)

    results = r.json()['items']

    video_ids = []
    for result in results:
        video_ids.append(result['id']['videoId'])

    youtube_url = f'https://www.youtube.com/watch?v={ video_ids[0] }'
    webbrowser.open_new(youtube_url)

    video_params = {
            'key' : YOUTUBE_API_KEY,
            'id' : ','.join(video_ids),
            'part' : 'snippet,contentDetails',
            'maxResults' : 20
        }
    r = requests.get(video_url, params=video_params)
    results = r.json()['items']
    for result in results:
        video_data = {
            'id' : result['id'],
            'url' : f'https://www.youtube.com/watch?v={ result["id"] }',
            'thumbnail' : result['snippet']['thumbnails']['high']['url'],
            'duration' : int(parse_duration(result['contentDetails']['duration']).total_seconds() // 60),
            'title' : result['snippet']['title'],
        }
        videos.append(video_data)
#========================================================================================================================
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
#========================================================================================================================

#========================================================================================================================
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')
#=======================================================================================================================
@app.route('/listing')
def Listing():
    search_url = 'https://www.googleapis.com/youtube/v3/search'
    video_url = 'https://www.googleapis.com/youtube/v3/videos'
    videos = []
    print("The max emotion is: ", max_emotion)
    search_params = {
            'key' : YOUTUBE_API_KEY,
            'q' : 'bollywood '+max_emotion+ ' songs',
            'part' : 'snippet',
            'maxResults' : 20,
            'type' : 'video'
        }

    r = requests.get(search_url, params=search_params)

    results = r.json()['items']

    video_ids = []
    for result in results:
        video_ids.append(result['id']['videoId'])

    video_params = {
            'key' : YOUTUBE_API_KEY,
            'id' : ','.join(video_ids),
            'part' : 'snippet,contentDetails',
            'maxResults' : 20
        }
    r = requests.get(video_url, params=video_params)
    results = r.json()['items']
    for result in results:
        video_data = {
            'id' : result['id'],
            'url' : f'https://www.youtube.com/watch?v={ result["id"] }',
            'thumbnail' : result['snippet']['thumbnails']['high']['url'],
            'duration' : int(parse_duration(result['contentDetails']['duration']).total_seconds() // 60),
            'title' : result['snippet']['title'],
        }
        videos.append(video_data)
    return render_template("index1.html",videos=videos)
#========================================================================================================================

if __name__ =="__main__":
    app.run(debug=True,port=8000)
