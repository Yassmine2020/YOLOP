# TODO:  ❌ apply mutiprocessing in the lane detection

# from multiprocessing import Pool

# def process_frame(frame_chunk):
#     # Perform lane detection on the frame chunk
#     # Return the results

# if __name__ == '__main__':
#     # Initialize video capture
#     video_capture = ...

#     # Number of processes to create
#     num_processes = 4

#     # Start pool of processes
#     with Pool(num_processes) as pool:
#         while True:
#             # Capture a frame
#             frame = video_capture.read()

#             # Divide the frame into chunks
#             frame_chunks = split_into_chunks(frame, num_processes)

#             # Process chunks in parallel
#             results = pool.map(process_frame, frame_chunks)

#             # Aggregate and display results
#             display_results(results)

# TO DO:
# from multiprocessing import Pool

# def process_frame(frame_info):
#     path, img, img_det, vid_cap, shapes, model, names, colors = frame_info

#     # Perform lane detection
#     ll_seg_out = model(img)
#     ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]

#     # Convert predictions to most likely class if it's multi-channel
#     if ll_predict.shape[1] > 1:
#         ll_predict_test = ll_predict.argmax(1)

#     ll_predict_img = ll_predict_test.squeeze().cpu().numpy()
#     ll_predict_img = (ll_predict_img * 255 / ll_predict_img.max()).astype(np.uint8)

#     img_det = extract_data(ll_predict_img)

#     # Further processing and visualization
#     if len(det):
#         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det.shape).round()
#         for *xyxy, conf, cls in reversed(det):
#             label_det_pred = f'{names[int(cls)]} {conf:.2f}'
#             plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=2)

#     return path, img_det, vid_cap

# if __name__ == '__main__':
#     # ... (Your existing code goes here)

#     # Run inference
#     t0 = time.time()

#     vid_path, vid_writer = None, None
#     img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
#     _ = model(img.half() if half else img) if device.type != 'cpu' else None
#     model.eval()

#     inf_time = AverageMeter()
#     nms_time = AverageMeter()

#     # Create a pool of processes
#     with Pool() as pool:
#         # Prepare frame info for parallel processing
#         frame_info_list = [(path, img, img_det, vid_cap, shapes, model, names, colors) for path, img, img_det, vid_cap, shapes in tqdm(dataset)]

#         # Use partial to pass additional arguments to process_frame function
#         process_frame_partial = partial(process_frame, model=model, names=names, colors=colors)

#         # Process frames in parallel
#         processed_frames = pool.map(process_frame_partial, frame_info_list)

#     # Display the results, save, or write to video as needed
#     for path, img_det, vid_cap in tqdm(processed_frames):
#         # ... (Your existing code goes here)

#     print('Results saved to %s' % Path(opt.save_dir))
#     print('Done. (%.3fs)' % (time.time() - t0))
#     print('inf: (%.4fs/frame)   nms: (%.4fs/frame)' % (inf_time.avg, nms_time.avg))



import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm

# 📛 TEST
from data_extraction.data_extraction import *
# 📛 TEST

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

##📛 TEST
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]

        # Convert predictions to most likely class if it's multi-channel
        if ll_predict.shape[1] > 1:  # Assuming the channel dimension is 1
            ll_predict_test = ll_predict.argmax(1)

        # Convert to a suitable image format (0-255 range, uint8 type)
            # to convert them to a NumPy array or use them with many other Python libraries, you need them on the CPU.
        ll_predict_img = ll_predict_test.squeeze().cpu().numpy()  # Squeeze [H,W] if it has a singleton dimension
        ll_predict_img = (ll_predict_img * 255 / ll_predict_img.max()).astype(np.uint8)  # Scale to 0-255

        img_det = extract_data(ll_predict_img)        

##📛 TEST

        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
