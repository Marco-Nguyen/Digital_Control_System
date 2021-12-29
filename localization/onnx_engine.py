import argparse
import os

import torch.backends.cudnn as cudnn

from utils import *

class ONNX_engine:
    def __init__(self):
        super().__init__()

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
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labelsq
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric()

        # # Directories
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        device = select_device(device)
        # half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        # if pt:
        #     model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        #     stride = int(model.stride.max())  # model stride
        #     names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        #     if half:
        #         model.half()  # to FP16
        #     if classify:  # second-stage classifier
        #         modelc = load_classifier(name='resnet50', n=2)  # initialize
        #         modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
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
        # else:  # TensorFlow models
        #     import tensorflow as tf
        #     if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
        #         def wrap_frozen_graph(gd, inputs, outputs):
        #             x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
        #             return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
        #                            tf.nest.map_structure(x.graph.as_graph_element, outputs))
        #
        #         graph_def = tf.Graph().as_graph_def()
        #         graph_def.ParseFromString(open(w, 'rb').read())
        #         frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        #     elif saved_model:
        #         model = tf.keras.models.load_model(w)
        #     elif tflite:
        #         if "edgetpu" in w:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
        #             import tflite_runtime.interpreter as tflri
        #             delegate = {'Linux': 'libedgetpu.so.1',  # install libedgetpu https://coral.ai/software/#edgetpu-runtime
        #                         'Darwin': 'libedgetpu.1.dylib',
        #                         'Windows': 'edgetpu.dll'}[platform.system()]
        #             interpreter = tflri.Interpreter(model_path=w, experimental_delegates=[tflri.load_delegate(delegate)])
        #         else:
        #             interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
        #         interpreter.allocate_tensors()  # allocate
        #         input_details = interpreter.get_input_details()  # inputs
        #         output_details = interpreter.get_output_details()  # outputs
        #         int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
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

        # Run inference
        # if pt and device.type != 'cpu':
        #     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, img, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            if onnx:
                img = img.astype('float32')
            # else:
            #     img = torch.from_numpy(img).to(device)
            #     img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # img = np.resize(img, (1, 3, 416, 416))

            # Inference
            # if pt:
            #     visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            #     pred = model(img, augment=augment, visualize=visualize)[0]
            if onnx:
                if dnn:
                    net.setInput(img)
                    pred = torch.tensor(net.forward())
                else:
                    pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
            # else:  # tensorflow model (tflite, pb, saved_model)
            #     imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            #     if pb:
            #         pred = frozen_func(x=tf.constant(imn)).numpy()
            #     elif saved_model:
            #         pred = model(imn, training=False).numpy()
            #     elif tflite:
            #         if int8:
            #             scale, zero_point = input_details[0]['quantization']
            #             imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
            #         interpreter.set_tensor(input_details[0]['index'], imn)
            #         interpreter.invoke()
            #         pred = interpreter.get_tensor(output_details[0]['index'])
            #         if int8:
            #             scale, zero_point = output_details[0]['quantization']
            #             pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            #     pred[..., 0] *= imgsz[1]  # x
            #     pred[..., 1] *= imgsz[0]  # y
            #     pred[..., 2] *= imgsz[1]  # w
            #     pred[..., 3] *= imgsz[0]  # h
            #     pred = torch.tensor(pred)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

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

                x_c = 0
                y_c = 0

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            x_c = (xyxy[0].numpy() + xyxy[2].numpy())/2
                            y_c = (xyxy[1].numpy() + xyxy[3].numpy())/2
                            # if save_crop:
                            #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({1/(t3 - t2):.3f}fps)')
                # t = tuple(x / seen * 1E3 for x in dt)
                # LOGGER.info(
                #     f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    cv2.circle(im0, (int(x_c), int(y_c)), 3, (0, 0, 255), cv2.FILLED)
                    cv2.imshow(str(p), im0)
                    key = cv2.waitKey(1)  # 1 millisecond
                    if key & 0xFF == 27:
                        break

                # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video' or 'stream'
                #         if vid_path[i] != save_path:  # new video
                #             vid_path[i] = save_path
                #             if isinstance(vid_writer[i], cv2.VideoWriter):
                #                 vid_writer[i].release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             else:  # stream
                #                 fps, w, h = 15, im0.shape[1], im0.shape[0]
                #                 save_path += '.mp4'
                #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vid_writer[i].write(im0)

                    # Print results
                    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
                    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, imgsz, imgsz)}' % t)
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

