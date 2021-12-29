from onnx_engine import *


"""RUN ONNX ENGINE"""
webcam_id = 0

if __name__ == '__main__':
    model = ONNX_engine()
    model.run(weights="path/to/weights.onnx", source=webcam_id)