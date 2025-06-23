from flask import Flask
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

import os
import json
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def get_file():
    return hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")

model = YOLOv10(get_file())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def doc_layout():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('no file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            det_res = model.predict(
                filename,   # Image to predict
                imgsz=1024,        # Prediction image size
                conf=0.2,          # Confidence threshold
                device="cpu"    # Device to use (e.g., 'cuda:0' or 'cpu')
            )
            names = det_res[0].names
            blocknames = [names[int(n)] for n in det_res[0].boxes.cls]
            xyxy = [a.tolist() for a in det_res[0].boxes.xyxy]
            res = [{"coords": y, "type": x} for x, y in zip(blocknames, xyxy)]
            return json.dumps(res)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
