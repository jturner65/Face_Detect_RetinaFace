#!/usr/bin/env python
"""
Utility functions used by script.py for face detection
"""
#imports

import cv2
import numpy as np
import os
from utils.mdl_config import cfg_mnet, cfg_re50
from models.retinaface import PriorBox
import torch
from typing import Any, Dict, List, Optional, Tuple



def check_file_or_dir(name : str, valid_ext : Optional[List] = []) -> str:
    """
    Check whether the passed string exists in the file system and is a valid file or a directory.

    Parameters
    ----------
    name : string
        Name of either file or directory to check.

    valid_ext : list of string, optional
        If supplied, delineates possible extensions for a file to have to be considered 
        valid. Case matters!

    Returns
    -------
    tuple of boolean
        IDX 0 is whether name is found in filesystem or not (and if an existing file, whether
        it has appropriate extension), IDX 1 is whether name is file (True) or directory (False)

    """
    if os.path.isfile(name):
        isValidFile = True
        # if list of extensions are provided, verify that found file has acceptable extension
        if len(valid_ext) > 0:
            isValidFile = os.path.splitext(name)[1] in valid_ext
        return (isValidFile, True)
    if os.path.isdir(name):
        return (True, False)
    return (False, False)


def get_net_config(modelName : str) -> Dict:
    """
    Get the RetinaFace config for the Resnet50 model.

    Parameters
    ----------
    modelName : str
        Name of modelName whose config we wish to retrieve - either "mobilenet" or "resnet"

    Returns
    -------
    dict : 
        Dictionary containing the various image configuration values used by the RetinaFace model for the
        loaded model

    """
    if "mobile025" in modelName.lower():
        # mobile0.25 config
        return cfg_mnet
    # otherwise return resnet config as default
    return cfg_re50


def build_thresh_dict(conf_threshold: float, nms_threshold: float)->Tuple:
    """
    Build a dictionary of confidence and nms thresholds, and a string concatenating both of them.

    Parameters
    ----------
    conf_threshold : float
        minimum confidence to consider a proposal valid
    nms_threshold : float
        minimum value for non-maximal suppression of overlapped proposals
    Returns
    -------
    tuple : dict, string
        tuple containing proposals and string of values concatenated

    """
    thresholds = {
        'conf':conf_threshold,
        'nms':nms_threshold
    }
    # string encoding thresholds to be used for model evaluations and image saving
    threshold_key = f'conf_{thresholds["conf"]:.2f}_nms_{thresholds["nms"]:.2f}'
    return thresholds, threshold_key
    

def find_faces_with_model(img: np.ndarray, 
                          model: Any, 
                          cfg: Dict,
                          rescale: float) -> Tuple:
    """
    This function executes a forward pass of the given model over the provided image.

    Parameters
    ----------
    img : ndarray
        Image possibly containing faces feed to the model
    model : pytorch model
        Trained RetinaFace model to use for face detection
    cfg : dict
        Configuration parameters of trained model
    rescale : float
        floating point value to correct for possible resizing of image
        when loaded, so that detection boxes' coordinates align with
        original image

    Returns
    -------
        Tuple of Tensors
            box proposal tensor, landms proposal tensors, scores collection
    """

    # retrieve ref to torch device
    device = torch.device('cpu')

    # This code from test_widerface.py script from RetinaFace source
    im_height, im_width, _ = img.shape
    # determine appropriate scale to use image
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # forward pass
    loc, conf, landms = model(img) 

    # build prior box and derive priors
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / rescale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / rescale
    landms = landms.cpu().numpy()
    return boxes, landms, scores

    # # ignore low scores
    # idxs = np.where(scores > thresholds['conf'])[0]
    # boxes = boxes[idxs]
    # landms = landms[idxs]
    # scores = scores[idxs]

    # # keep top-K before NMS
    # order = scores.argsort()[::-1]
    # boxes = boxes[order]
    # landms = landms[order]
    # scores = scores[order]

    # dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    # # perform NMS - non maximal suppression. Keep only those overlapping proposals that meet
    # # given threshold
    # keep = py_cpu_nms(dets, thresholds['nms'])
    # dets = dets[keep, :]
    # landms = landms[keep]
    # # rebuild detections structure to contain landms values (facial feature location points)
    # dets = np.concatenate((dets, landms), axis=1)

    # return dets

def filter_results_from_net(thresholds: Dict,
                            boxes: Any,
                            landms: Any,
                            scores: Any):
    """
    This will filter the result values from the net based on the passed threshold values
   
 
    Parameters
    ----------  
   
    thresholds : dict of float
        'conf' is confidence threshold of detections [0-1]
        'nms' is non-maximal suppression threshold for overlaps [0-1]

    Returns
    -------
        ndarray
            Array structure holding detection proposals
   
    """

    # ignore low scores
    idxs = np.where(scores > thresholds['conf'])[0]
    boxes = boxes[idxs]
    landms = landms[idxs]
    scores = scores[idxs]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    # perform NMS - non maximal suppression. Keep only those overlapping proposals that meet
    # given threshold
    keep = py_cpu_nms(dets, thresholds['nms'])
    dets = dets[keep, :]
    landms = landms[keep]
    # rebuild detections structure to contain landms values (facial feature location points)
    dets = np.concatenate((dets, landms), axis=1)

    return dets



def get_filenames_from_subdir(src_dir: str, 
                              valid_ext : Optional[List] = [], 
                              debug: Optional[bool] = False) -> List:

    """
    This function walks a given source directory for all files and returns a list 
    of path-qualified filenames.

    Parameters
    ----------
    src_dir : string
        Directory to walk

    valid_ext : list of string, optional
        If supplied, delineates possible extensions for a file to have to be considered 
        valid. Case matters!

    debug : boolean, optional
        Whether to display debug information - count of files found

    Returns
    -------
    List 
        path-qualified filenames
    """
    res_list = []
    found_count = 0

    # validate extensions
    if len(valid_ext) > 0:            
        for path, _, files in os.walk(src_dir):
            for fname in files:
                isValidFile = os.path.splitext(fname)[1] in valid_ext
                if isValidFile:
                    found_count += 1
                    res_list.append(os.path.join(path, fname))

    else: 
        for path, _, files in os.walk(src_dir):
            for fname in files:
                found_count += 1
                res_list.append(os.path.join(path, fname))
    if debug:
        print(
            f"get_files_from_subdir : Found and matched {len(res_list)} files in {src_dir}"
        )   
    return res_list

def get_num_faces_expected(image_path : str) -> int :
    """
    Parse given file path to get expected number of faces to be found in image based on filename.
    Name is expected to be of format `<x>_faces_#.ext` where `x` is the number of expected faces.

    Parameters
    ----------
    image_path : string
        Path and filename for the image to load

    Returns
    -------
    int 
        Number of faces expected to be found in image
    """
    num_faces_str = os.path.basename(image_path).split("_faces")[0].strip()
    return int(num_faces_str)

def load_img(image_path : str, do_resize : Optional[bool] = False) -> Tuple : 
    """
    Load image from passed image path and return as both processed numpy array and unprocessed 
    matrix

    Parameters
    ----------
    image_path : string
        Path and filename for the image to load

    do_resize : boolean, optional
        Resize/rescale image. Default is false

    Returns
    -------
    tuple of numpy float array, cv2.Mat, float
        Loaded image as numpy array, loaded image as original matrix, value of resize 
        (to be used to scale after fwd pass)
    """
    try: 
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    except: 
        print(f"Attempting to load {image_path} as an image failed. Aborting.")
        exit()
    img = np.float32(img_raw)
    resize = 1
    if do_resize:
        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    return img, img_raw, resize

def get_res_dir(all_res_dict: dict, subdir: str):
    # build the name of the subdirectory to put eval results in
    res_dir = all_res_dict['res_dir']
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    res_dir = os.path.join(res_dir,subdir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    return res_dir    


def save_eval_csv(all_res_dict: dict):
    f"""
    Save a csv with a single image's expected vs detected results for a range of
    confidence and nms threshold values.

    Parameters
    ----------
    all_res_dict : dict
        Dictionary holding two entries :
            'res_dir' : string
                Relative path to desired destination directory.
            'results : dict
                Holds per-image dictionaries of detection results for various
                confidence and nms threshold values
                Each dictionary is:
                    Dictionary containing all pertinent information for a particular image.
                    'image_path' : string
                        Path and filename of original image. The filename for the result image 
                        will be taken from this.
                    'results' : dict
                        Dictionary containing dicts of counts for each tested combination of 
                        confidence and nms thresholds, keyed by string encoding confidence 
                        threshold and nms threshold used to derive detection proposals
                            Key-value pairs within each sub-dict :
                                'thresholds': dict of floats
                                    'conf' and 'nms' values
                                'num_detections' : int
                                    number of detections with given threshold
    
    Returns
    -------
    None

    """
    import csv
    # build list of dicts of res
    main_output_list = []
    res_dir = get_res_dir(all_res_dict,'eval_data_res/')
    main_output_file_name = f'{res_dir}All_images_eval_results.csv'
    # fields in csvs
    fieldnames = ['Image','Entry', 'Confidence Thresh', 'NMS Thresh', 'Expected Count', 'Detected Count', 'Matched Dets']
    
    for _, res_dict in sorted(all_res_dict["results"].items()):
        img_filename = res_dict['image_path']
        split_ext = os.path.splitext(os.path.basename(img_filename))
        base_img_name = split_ext[0]
        #output file name
        output_file_name = f'{res_dir}{base_img_name}_results.csv'

        # Number of expected faces for this file, based on file name
        num_faces_expected = get_num_faces_expected(img_filename)  
        # get all results for this image
        res_values_dict = res_dict['results'] 

        # build list of dicts of res
        output_list = []
        for k,v in sorted(res_values_dict.items()):
            tmp_dict = {}
            tmp_dict['Image'] = base_img_name
            tmp_dict['Confidence Thresh'] = v['thresholds']['conf']
            tmp_dict['NMS Thresh'] = v['thresholds']['nms']
            tmp_dict['Entry'] = k
            tmp_dict['Expected Count'] = num_faces_expected
            tmp_dict['Detected Count'] = v["num_detections"]
            tmp_dict['Matched Dets'] = (num_faces_expected == v["num_detections"])
            output_list.append(tmp_dict)
            main_output_list.append(dict(tmp_dict))

        # Save per image report
        with open(output_file_name,'w', encoding='UTF8', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(output_list)

        print(f'Saving csv data for {base_img_name} to {output_file_name}')
    # Save all images report
    with open(main_output_file_name,'w', encoding='UTF8', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(main_output_list)


def plot_all_results(all_res_dict: dict):
    f"""
    Plot the performance of varying the confidence and non-maximal suppression threshold 
    values, aggregated across all images.

    Parameters
    ----------
    all_res_dict : dict
        Dictionary holding two entries :
            'res_dir' : string
                Relative path to desired destination directory.
            'results : dict
                Holds per-image dictionaries of detection results for various
                confidence and nms threshold values
                Each dictionary is:
                    Dictionary containing all pertinent information for a particular image.
                    'image_path' : string
                        Path and filename of original image. The filename for the result image 
                        will be taken from this.
                    'results' : dict
                        Dictionary containing dicts of counts for each tested combination of 
                        confidence and nms thresholds, keyed by string encoding confidence 
                        threshold and nms threshold used to derive detection proposals
                            Key-value pairs within each sub-dict :
                                'thresholds': dict of floats
                                    'conf' and 'nms' values
                                'num_detections' : int
                                    number of detections with given threshold
    
    Returns
    -------
    None

    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from collections import OrderedDict
    except:
        print('Matplotlib not found, so unable to plot results. Aborting.')
        return
    res_dir = get_res_dir(all_res_dict,'eval_data_plots/')
    for _, res_dict in sorted(all_res_dict["results"].items()):
        img_filename = res_dict['image_path']
        split_ext = os.path.splitext(os.path.basename(img_filename))
        base_img_name = split_ext[0]
        plot_file_name = f'{res_dir}{base_img_name}_results.jpg'
        # Number of expected faces for this file, based on file name
        num_faces_expected = get_num_faces_expected(img_filename)  
        # get all results for this image
        res_values_dict = res_dict['results'] 

        xDict = OrderedDict()
        yDict = OrderedDict()
        tmpZVals = {}
        for _,v in sorted(res_values_dict.items()):
            valsDict = v['thresholds']
            x = valsDict['conf']
            y = valsDict['nms']
            xDict[x] = x
            yDict[y] = y
            # save # incorrect for this combination
            tmpZVals[(x,y)] = abs(num_faces_expected - v["num_detections"])

        zValues = np.empty(shape=(len(yDict),len(xDict)))
        x = 0
        for xVal in xDict:
            y = 0
            for yVal in yDict:
                incorrect_val = tmpZVals[(xVal, yVal)]
                zValues[y,x] = incorrect_val
                y += 1
            x += 1
        xValues = list(xDict.keys())
        yValues = list(yDict.keys())

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(f'{base_img_name} : Number incorrect vs. Confidence and NMS Thresholds')
        xValues, yValues,  = np.meshgrid(xValues, yValues)

        # Plot the surface.
        surf = ax.plot_surface(xValues, yValues, zValues, cmap=cm.coolwarm, antialiased=True,linewidth=0.3,
                       alpha = 0.8, edgecolor = 'k')
        ax.set(xlabel = "Confidence Threshold", ylabel = "NMS Threshold", zlabel = "# Incorrect")

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        print(f'Saving plot for {base_img_name} to {plot_file_name}')
        ax.invert_xaxis()
        plt.savefig(plot_file_name, dpi=80)
        plt.close()



def save_annotated_image(dets : np.ndarray, 
                         img_raw : np.ndarray, 
                         res_dir : str, 
                         image_path : str, 
                         threshold_str : str) -> None:
    """
    Save a copy of the image having been tested with annotations around face 
    proposals and confidence values for each proposal.        

    Parameters
    ----------
    dets : ndarray
        The detection data holding the bounds and confidence of each detection. 
        Idxs 5:14 also hold facial feature proposals from Retinaface

    img_raw : ndarray
        The loaded, unmodified source image

    res_dir : string
        Relative path to desired destination directory.

    image_path : string
        Path and filename of original image. The filename for the result image 
        will be taken from this.

    threshold_str : string
        Holds string encoding confidence threshold and nms threshold used to derive
        detection proposals. Used for file name.
    
    Returns
    -------
    None

    """
    split_ext = os.path.splitext(os.path.basename(image_path))
    img_name = f'{split_ext[0]}_{threshold_str}{split_ext[1]}'
    #print(f'\tSaving file named `{img_name}`')
    for b in dets:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # save image
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    name = res_dir + img_name
    cv2.imwrite(name, img_raw)


# --------------------------------------------------------
# Code below comes from RetinaFace source


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def py_cpu_nms(dets, thresh):
    """
    Pure Python NMS baseline. From RetinaFace util
    
    Returns
    -------
        list    
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# From RetinaFace source code
# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Returns
    -------
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """
    Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Returns
    -------
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms
