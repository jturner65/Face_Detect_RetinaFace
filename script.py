#!/usr/bin/env python
"""
This program uses pre-trained RetinaFace models to detect faces in provided images. 
For more on RetinaFace see : https://github.com/biubug6/Pytorch_Retinaface
Both Resnet50 and Mobilenet0.25 based pretrained models are supported. 
"""

# imports
import argparse
import torch

# module import
import utils.utils as ut


# Dict holding pertinent info for available pretrained RetinaFace nets
AVAILABLE_NETS_INFO = {
    'name': {
        'resnet50': 'Resnet50', 
        'mobile025':'Mobilenet0.25'},
    'model_path': {
        'resnet50': './models/Resnet50_Final_model.pth', 
        'mobile025':'./models/mobilenet0.25_Final_model.pth'},
    'results_path': {
        'resnet50': './results_resnet/', 
        'mobile025':'./results_mobile025/'},
    'dflt_conf_thresh': {
        'resnet50': 0.8, 
        'mobile025':0.7},
    'dflt_nms_thresh': {
        'resnet50': 0.4, 
        'mobile025':0.4}    
}

# Valid/expected file extensions for image load
VALID_EXT = ['.jpg','.png','.JPG','.PNG']


def main():
    parser = argparse.ArgumentParser(description='Count faces in images')
    parser.add_argument('file_or_dir_name', type=str, 
                        help='Either the name of an image to find and count faces \
                            in, or the name of a directory containing a list of such \
                            images. Should be .jpg or .png extension.')
    parser.add_argument('-e','--eval', action='store_true', default=False, 
                        help='Evaluate results for ranges of confidence and non-maximal \
                            threshold values [0-1) and save the results as csv files \
                            and, if matplotlib is installed, via 3d plots.')
    parser.add_argument('-n','--network', default='resnet50', 
                        choices=['resnet50','mobile025'], 
                        help='Backbone network to use for detection.')
    parser.add_argument('-s', '--save_image', action='store_true', default=False, 
                        help='Save annotated images showing detection results.')
    parser.add_argument('-v', '--verbose_msgs', action='store_true', default=False, 
                        help='Print verbose results to the console.')


    args = parser.parse_args()
    
    # Map parser args
    # Should be either a file name or a directory name
    file_or_dir_name = args.file_or_dir_name

    # Determine whether requested file or dir exists, what it was, and if it is of valid format
    file_or_dir_res = ut.check_file_or_dir(file_or_dir_name, VALID_EXT)
    if not file_or_dir_res[0]:
        if not file_or_dir_res[1]:
            print(f'{file_or_dir_name} is not found as either a file or a directory.  Aborting.')
        else :
            print(f'{file_or_dir_name} was found as a file but has unsupported extension.  Aborting.')
        exit()

    # Pretrained network to use - this is a key to AVAILABLE_NETS_INFO dict's subdicts
    net_to_use = args.network

    found = False
    # Verify model exists, otherwise default to an existing model or exit gracefully
    while not found :
        model_path_to_use = AVAILABLE_NETS_INFO['model_path'][net_to_use]
        model_found_res = ut.check_file_or_dir(model_path_to_use)
        if not model_found_res[0]:
            if net_to_use == 'resnet50':
                print(f'"Resnet50_Final_model.pth" not found in "models/" directory.  \
                Attempting to load "mobilenet0.25_Final_model.pth" instead. This model \
                does not perform as well.')
                # Try again with mobile025
                net_to_use = 'mobile025'
            else:
                print(f'No valid network models found in "models/" directory.  Aborting.')
                exit()
        else:
            print(f"Found {model_path_to_use} model.")
            found = True
    # Whether or not we should evaluate the results
    eval_results = args.eval
    # Whether or not to save annotated images
    save_image = args.save_image
    # verbose console output
    verbose_msgs = args.verbose_msgs


    # Only consuming the model, no need for gradients
    torch.set_grad_enabled(False)

    # Build list containing either a single image file name or all file names within the passed directory
    if file_or_dir_res[1] : 
        image_file_list = [file_or_dir_name]    
    else :
        image_file_list = ut.get_filenames_from_subdir(file_or_dir_name, VALID_EXT)

    # Load the pretrained model trained by RetinaFace    
    model = torch.load(model_path_to_use)  
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    # get RetinaFace image/model config
    cfg = ut.get_net_config(net_to_use)

    results_path = AVAILABLE_NETS_INFO['results_path'][net_to_use]

    # set confidence and nms values based on defaults for net arch
    conf_range = [AVAILABLE_NETS_INFO['dflt_conf_thresh'][net_to_use]]
    nms_range = [AVAILABLE_NETS_INFO['dflt_nms_thresh'][net_to_use]]

    # Build threshold dict and threshold key for entire model
    thresholds, threshold_key = ut.build_thresh_dict(conf_range[0], nms_range[0])

    if eval_results:
        # build ranges for calculation
        from numpy import arange
        conf_range = arange(0.05, 1.0, 0.05)
        nms_range = arange(0.05, 1.0, 0.05)

    # if not verbose then show expected results string
    show_res_str = not verbose_msgs
 
    # Only used for result eval
    thresh_all_results = {}
    thresh_all_results['res_dir'] = results_path   
    thresh_all_results['results'] = {} 

    # load each image to check, resize them to match retina face testing protocol, and check
    for img_filename in sorted(image_file_list):
        # number of faces expected based on filename
        num_faces_expected = ut.get_num_faces_expected(img_filename) 

        # load image, and get a copy of the image for calcs
        img, img_raw, resize = ut.load_img(img_filename)

        # Get raw proposals and information from net for image
        boxes, landms, scores = ut.find_faces_with_model(img, model, cfg, resize)


        # If evaluating results, sweep through threshold values and save results to csv
        if eval_results : 
            thresh_results_dict = {}
            thresh_results_dict['image_path'] = img_filename
            thresh_results_dict['results'] = {}
            # if not evaluating, the following loops each only have a single value
            for conf_threshold in conf_range:
                for nms_threshold in nms_range:

                    # Build threshold dict and threshold key for given threshold values
                    thresholds, threshold_key = ut.build_thresh_dict(conf_threshold, nms_threshold)
                    
                    # filter results based on thresholds
                    dets = ut.filter_results_from_net(thresholds, boxes, landms, scores)
                    num_detections = len(dets)        

                    tmp_dict = {}
                    tmp_dict['thresholds'] = thresholds
                    tmp_dict['num_detections'] = num_detections
                    thresh_results_dict['results'][threshold_key] = tmp_dict
                    if verbose_msgs:
                        # expanded output
                        expected_equals = num_detections == num_faces_expected
                        if expected_equals :
                            print(f'\t\t{img_filename} : # faces found : {num_detections} as expected')
                        else :
                            print(f'{img_filename} : # faces found : {num_detections} | # faces expected {num_faces_expected}')

            print(f'Finished collecting data for image {img_filename}')
            thresh_all_results['results'][img_filename] = thresh_results_dict
        else :
            # Not evaluating results, single pass using specified threshold
            
            # filter results based on thresholds
            dets = ut.filter_results_from_net(thresholds, boxes, landms, scores)
            num_detections = len(dets)

            if show_res_str :
                # show primary requirement to console
                print(f'{img_filename}\t{num_detections}')

            if verbose_msgs:
                # expanded output
                expected_equals = num_detections == num_faces_expected
                if expected_equals :
                    print(f'\t\t{img_filename} : # faces found : {num_detections} as expected')
                else :
                    print(f'{img_filename} : # faces found : {num_detections} | # faces expected {num_faces_expected}')


            if save_image:
                # save annotated images in results directory
                ut.save_annotated_image(dets, img_raw, results_path, img_filename, threshold_key)

        
    if eval_results:
        # save evaluations of all images
        ut.save_eval_csv(thresh_all_results)
        # plot results
        ut.plot_all_results(thresh_all_results)

 
if __name__ == '__main__':
    main()
