from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import cv2     # for capturing videos
import math
from keras_preprocessing import image
import pandas as pd
import numpy as np
from tensorflow.keras.applications import InceptionV3


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=(224,224,3))
lstm_model = keras.models.load_model('nextword_model.h5')
class_df = pd.read_csv(os.path.join(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], 'video_classes.csv')
classes = class_df[video_class]

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        return "File has been uploaded."
    return render_template('index.html', form=form)

@app.route('/predict', methods=['POST'])
def checkImage():
    if request.method == 'POST':
        f = request.files['file']
        file_name = f.filename
        print(file_name)
        t = request.form.get("query")
        f.save(secure_filename(f.filename))
        
        return "File has been uploaded"

    # Take in a video as input
    input_video = file_name

    # Frame the video
    video_frames_arr = []

    videoFile = input_video
    vid_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(f.filename))  # capturing the video from the given path
    
    cap = cv2.VideoCapture(vid_path)
    print(cap.isOpened())
    frameRate = cap.get(5)  # Returns the frame rate or fps of the video
    currFrame = 0
    x = 1

    while(cap.isOpened()):
        frameId = cap.get(1) # Returns the current frame index/ number to be captured next
        ret, frame = cap.read()
        if ret != True:
            break

        if frameId % math.floor(frameRate) == 0:
            # storing the frames in train directory
            img = videoFile.split('_')[1] + "_frame" + str(currFrame) + ".jpg"
            filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], img)
            currFrame += 1
            
            video_frames_arr.append(filename)
        

    cap.release()
    print("Done!")

    video_frames_arr

    video_frames = []

    # Load the images
    for i in video_frames_arr:
        img = i
        img = image.load_img(img, target_size=(224,224,3))
        img_arr = image.img_to_array(img)
        img_arr /= 255.0
        video_frames.append(img_arr)

    video_frames = np.array(video_frames)
    video_frames.shape


    video_frames = base_model.predict(video_frames)
    video_frames.shape

    video_frames = video_frames.reshape(video_frames.shape[0], 5*5*2048)

    max = video_frames.max()
    video_frames = video_frames/max

    video_frames.shape

    # Make prediction
    predicted_classes = model.predict(video_frames)

    highest_predicted_classes = []
    for i in predicted_classes:
        yhat = np.argmax(i, axis=-1) 
        class_name = classes[yhat]
        highest_predicted_classes.append(class_name)

    highest_predicted_classes

    input_query = 'PullUps'

    for k in range(len(predicted_classes)):
        if input_query == predicted_classes[k]:
            found_img = video_frames_arr[k]
            loaded_img = image.load_img(found_img, target_size=(224,224,3))
            loaded_img



if __name__ == '__main__':
    app.run(debug=True)