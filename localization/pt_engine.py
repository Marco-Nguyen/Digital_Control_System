import cv2
import torch
from PIL import Image
import numpy as np
import time

webcam_id = 0
x_center = []
y_center = []

class Yolov5:
    def __init__(self, model_path="path/to/weight.pt"):
        super().__init__()
        self.model = torch.hub.load("ultralytics/yolov5", 'custom', path=model_path)
        self.setup_params()

    def setup_params(self):
        self.imgsz = 416
        self.model.conf = 0.7  # NMS confidence threshold
        self.model.iou = 0.5  # NMS IoU threshold
        self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1  # maximum number of detections per image
        self.x_center = []
        self.y_center = []

    def detect(self, frame):
        # time_start = time.time()
        # self.setup_params()
        results = self.model(frame, size=416)
        # results.print()
        results.save()
        # self.model.conf = 0.7  # NMS confidence threshold
        # self.model.iou = 0.75  # NMS IoU threshold
        # self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
        # self.model.multi_label = False  # NMS multiple labels per box
        # self.model.max_det = 1  # maximum number of detections per image
        # print('\n', results.xyxy)

        # x_min = results.xyxy[0][0][0].numpy()
        # y_min = results.xyxy[0][0][1].numpy()
        # x_max = results.xyxy[0][0][2].numpy()
        # y_max = results.xyxy[0][0][3].numpy()
        # w = x_max - x_min
        # h = y_max - y_min
        # result = [x_min, y_min, x_max, y_max]
        # # Calculate mean + rms
        # xc = (results.xyxy[0][0][0].numpy() + results.xyxy[0][0][2].numpy())/2
        # yc = (results.xyxy[0][0][1].numpy() + results.xyxy[0][0][3].numpy())/2
        # x_center.append(xc)
        # y_center.append(yc)
        # xc_mean = np.mean(xc)
        # yc_mean = np.mean(yc)
        #
        # xc_var = np.float(np.sqrt((np.sum(np.square(xc - xc_mean))) / len(x_center)))
        # yc_var = np.float(np.sqrt((np.sum(np.square(yc - yc_mean))) / len(y_center)))
        #
        # print([xc_mean, yc_mean, xc, yc, xc_var, yc_var])
        #
        # file = open("results-yolov5n.txt", "a")
        # file.writelines([str(x_min) + " ", str(y_min) + " ", str(w) + " ", str(h) + " ", "\n"])
        # fps = 1/(time.time()-time_start)
        # print(fps)


"""RUN PT ENGINE"""
weight_path = r"D:\Python\Pycharm\AI-in-agriculture\weights\Solar-car\new-data\yolov5s\best.pt"
cap = cv2.VideoCapture(webcam_id)

if __name__ == '__main__':
    t0 = time.time()
    solar_car_detector = Yolov5(model_path=weight_path)
    print("Load model time:", time.time() - t0)
    while True:
        success, frame = cap.read()
        t1 = time.time()
        solar_car_detector.detect(frame)
        print("Detect FPS:", 1/(time.time() - t1))
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

