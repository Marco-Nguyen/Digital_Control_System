# Digital Control System Project

Description: <br/>
This module is about object tracking based on Yolov5 algorithm, design by Tai Hoang and Con Muc :v <br/>

## Usage

### Inference:

1. Specify “weights_path” (onnx weight) and set “view_img” to True if you want to view inference result
2. Specify <image_path> and <video_path> to run

## Yolov5

1. Preapre dataset: get from [my kaggle dataset](https://kaggle.com/danielaltanwing/solar-car-dataset)
2. Run inference to generate more data:

```
$ python path/to/yolov5/detect.py
--weights path/to/weights.pt
--img <image_size>
--conf <confidence_score>
--source path/to/source
--save-txt
```

Notice:

- source: can be 0, 1 for webcam or image path
- generated data can be slightly different from desired, can use [Roboflow](https://app.roboflow.com/) to change it manually

3. Training:

```python
$ python train.py --img 416 --batch 16 --epochs 100 --patience 25 --weights path/to/weights.pt --cache --cfg path/to/yolov5/models/ yolov5n.yaml --data path/to/data.yaml
```

```python
$ python train.py --img 416 --batch 16 --epochs 100 --patience 25 --weights yolov5s.pt --cache --cfg path/to/yolov5/models/ yolov5n.yaml --data path/to/data.yaml
```

Ref: [Yolov5](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) <br/>
Notebook: [Google Colab](https://colab.research.google.com/drive/1H1u6qLbcO9R9IHFtKdIHDZQc1rxV1jno?usp=sharing)

Notice:

- Patience: early stopping after … epochs
- Weights: can be “yolov5s.pt” if training from scratch (second command); or custom weights (link to download pt weights: ) (first command)
- Cfg: choose yolov5n or yolov5s (weights should be the same model if using custom weights)

4. Export to onnx:

```python
$ python export.py --weights path/to/best.pt --include onnx --dynamic --img 416 --data path/to/data.yaml
```

## Github repo

[link](https://github.com/Marco-Nguyen/Digital_Control_System)
