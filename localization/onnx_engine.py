import argparse
import os

import torch.backends.cudnn as cudnn

from utils import *

class ONNX_engine:
    def __init__(self):
        super().__init__()
        self.x_c = 0
        self.y_c = 0

    @torch.no_grad()
    def run(self,
            weights='path/to/weights',  # model.pt path(s)
            source=0,  # file/dir/URL/glob, 0 for webcam
            imgsz=416,  # inference size (pixels)
            conf_thres=0.7,  # confidence threshold
            iou_thres=0.5,  # NMS IOU threshold
            max_det=1,  # maximum detections per image
            device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            update=False,  # update all models
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric()

        # Initialize
        device = select_device(device)

        # Load model
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        if onnx:
            if dnn:
                ## opencv-python>=4.5.4
                # check_requirements(('opencv-python>=4.5.4',))
                net = cv2.dnn.readNetFromONNX(w)
            else:
                ## onnxruntime for CPU
                # check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
                import onnxruntime
                session = onnxruntime.InferenceSession(w, None)

        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            # view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=onnx)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=onnx)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        dt, seen = [0.0, 0.0, 0.0], 0
        for path, img, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            if onnx:
                img = img.astype('float32')

            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            if onnx:
                if dnn:
                    net.setInput(img)
                    pred = torch.tensor(net.forward())
                else:
                    pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            self.x_c = (xyxy[0].numpy() + xyxy[2].numpy())/2
                            self.y_c = (xyxy[1].numpy() + xyxy[3].numpy())/2
                            print("Bounding Box Center: ({}, {})".format(self.x_c, self.y_c))
                            # if save_crop:
                            #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. ({1/(t3 - t2):.3f}fps)')
                # t = tuple(x / seen * 1E3 for x in dt)
                # LOGGER.info(
                #     f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    cv2.circle(im0, (int(self.x_c), int(self.y_c)), 3, (0, 0, 255), cv2.FILLED)
                    key = cv2.waitKey(1)  # 1 millisecond
                    if key & 0xFF == 27:
                        view_img = False
                    else:
                        cv2.imshow(str(p), im0)

                    # Print results
                    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
                    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, imgsz, imgsz)}' % t)

        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


"""RUN ONNX ENGINE"""
# weights_path = r"best.onnx"
webcam_id = 0
image_path = r"path/to/image.jpg"

weights_path = r"path/to/best.onnx"

if __name__ == '__main__':
    model = ONNX_engine()
    print(model.x_c, model.y_c)
    model.run(weights=weights_path, source=image_path, view_img=True)