from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer



def check_keys(model, pretrained_state_dict):
    """Validate that the pre-trained model has all the parameters needed by the model"""
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """Remove old storage format prefix
    
    Old style model is stored with all names of parameters sharing common prefix 'module.'
    """

    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    """Loads the pretrained model from disk"""

    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect(img, device, timers = None):
    """Detects faces in an image.

    Detects face bounding boxes in an image using the FaceBoxes algorithm.
  
    Parameters: 
    img (numpy.ndarray): image as numpy array of type float32
    device (torch.device): device to run the inference on
    timers (dict): timers to use for timing each individual phase

    Returns: 
    numpy.ndarray: Detected face bounding boxes and their scores. 
    Each bouding box (i) is encoded as follows
        xmin = dets[i, 0]
        ymin = dets[i, 1]
        xmax = dets[i, 2]
        ymax = dets[i, 3]
        score = dets[i, 4]
    """
    if timers:
        timers['prep'].tic()
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    if timers:
        timers['prep'].toc()
        timers['forward_pass'].tic()
    out = net(img)  # forward pass
    if timers:
        timers['forward_pass'].toc()
        timers['filter'].tic()
    priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
    priors = priorbox.forward()
    priors = priors.to(device)
    loc, conf, _ = out
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #keep = py_cpu_nms(dets, args.nms_threshold)
    keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    
    if timers:
        timers['filter'].toc()

    return dets


if __name__ == '__main__':
    # Parse command line args
    parser = argparse.ArgumentParser(description='FaceBoxes')

    parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    args = parser.parse_args()

    # Initialize the network and the model
    torch.set_grad_enabled(False)
    net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # Init timers
    _t = {'capture': Timer(), 'prep': Timer(), 'forward_pass': Timer(), 'filter': Timer(), 'draw': Timer()}

    # Init live display
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 640,480)

    # Init camera video capture
    cap = cv2.VideoCapture(0)

    # Acquisition loop
    while True:
        # Read image from the camera
        _t['capture'].tic()
        ret, raw_img = cap.read()
        img = np.float32(raw_img)
        _t['capture'].toc()
        
        # Run face detector
        dets = detect(img, device, _t)

        # Draw bounding boxes
        _t['draw'].tic()
        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            score = dets[k, 4]
            if score<0.5: continue
            cv2.rectangle(raw_img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),1)

        # Display image
        cv2.imshow('image',raw_img)
        _t['draw'].toc()

        # Print timing stats
        print('capture: {:.1f}ms prep: {:.1f}ms inference: {:.1f}ms filter: {:.1f}ms draw: {:.1f}ms'.format(
                _t['capture'].average_time*1000, 
                _t['prep'].average_time*1000, 
                _t['forward_pass'].average_time*1000, 
                _t['filter'].average_time*1000,
                _t['draw'].average_time*1000
                ))
 
        # Wait for key q to quit loop
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
