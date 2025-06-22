from flask import Flask
from doclayout_yolo import YOLOv10
model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")


app = Flask(__name__)

@app.route("/")
def hello_world():
    filename = "dvurog_p73.png"
    det_res = model.predict(
        filename,   # Image to predict
        imgsz=1024,        # Prediction image size
        conf=0.2,          # Confidence threshold
        device="cpu"    # Device to use (e.g., 'cuda:0' or 'cpu')
    )

    return "<p>" + str(det_res[0].boxes.xyxy) + "</p>"
