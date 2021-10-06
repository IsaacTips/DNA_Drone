import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from utils.datasets import letterbox

import os
import sys
import glob
import numpy as np

from eval_utils import mAP

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def read_txt(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    #######################################################################################################
    # GYNetworks
    #######################################################################################################
    def calculate_parameters(model):
        return sum(param.numel() for param in model.parameters())/1000000.0
    print('[i] Total Params: %.2fM'%(calculate_parameters(model)))

    root_dir = opt.domain


    image_paths = glob.glob(root_dir + '/images/*')

    inference_times = []

    f = open('./prediction.csv', 'w')

    for index, image_path in enumerate(image_paths):
        sys.stdout.write(f'\r[{index + 1}/{len(image_paths)}] - {image_path}')
        sys.stdout.flush()

        image_name = os.path.basename(image_path)

        img0 = cv2.imread(image_path)

        st = time.time()

        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = img.to(device)
        
        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # process prediction
        # preds = [
        #     # person
        #     []
        # ]

        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, class_index in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line = (class_index, *xywh, conf)

                    xyxy = torch.tensor(xyxy).view(1, 4).view(-1).numpy().astype(np.int32).tolist()  # normalized xywh

                    class_index = int(class_index.item())

                    confidence = conf.item()
                    xmin, ymin, xmax, ymax = xyxy

                    # preds[class_index].append([xmin, ymin, xmax, ymax, confidence])
                    f.write('{},{:.0f},{:.0f},{:.0f},{:.0f},{:.2f}\n'.format(image_name, xmin, ymin, xmax, ymax, confidence))

        inference_time = int((time.time() - st) * 1000)
        inference_times.append(inference_time)

        # process gt
        # height, width = img0.shape[:2]
        # gt = []

        # label_path = image_path.replace('/images', '/labels').replace('.jpg', '.txt')

        # for data in open(label_path):
        #     class_index, cx, cy, w, h = data.strip().split(' ')

        #     class_index = int(class_index)
        #     cx = int(float(cx) * width)
        #     cy = int(float(cy) * height)
        #     w = int(float(w) * width)
        #     h = int(float(h) * height)

        #     xmin = cx - w // 2
        #     ymin = cy - h // 2
        #     xmax = cx + w // 2
        #     ymax = cy + h // 2

        #     gt.append([xmin, ymin, xmax, ymax, class_index])

        # visualize
        # for data in preds[0]:
        #     xmin, ymin, xmax, ymax = data[:4]
        #     cv2.rectangle(img0, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # for data in gt:
        #     xmin, ymin, xmax, ymax = data[:4]
        #     cv2.rectangle(img0, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # cv2.imshow('Demo', img0)
        # cv2.waitKey(0)
        
        # evaluator.add(preds, gt)
    
    #######################################################################################################
    # _, aps = evaluator.calc_mean_ap()
    # ap = aps[0]

    inference_time = int(np.mean(inference_times[10:]))

    print()
    # print('# AP@0.50 = {:.2f}%'.format(ap * 100))
    print('# Inference time per an image = {}ms'.format(inference_time))
    print()

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--domain', type=str, default='{HOME}/datasets_rw/challenge-dataset/public'.format(HOME=os.environ["HOME"]))

    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
