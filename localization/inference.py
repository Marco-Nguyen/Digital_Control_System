from onnx_engine import *


"""RUN ONNX ENGINE"""
weights_path = r"best.onnx"
webcam_id = 0

if __name__ == '__main__':
    model = ONNX_engine()
    model.run(weights=weights_path, source=webcam_id)