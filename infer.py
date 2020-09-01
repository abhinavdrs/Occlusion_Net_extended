from lib.config import cfg
from lib.predictor import COCODemo
import cv2
import argparse
import distutils.util
import csv
import os
import shutil
import matplotlib.path as mplPath
import torch
import numpy as np
import datetime
import json
import time


def parse_args():
    parser = argparse.ArgumentParser(description='evaluation on zensors data')

    parser.add_argument(
        '-cfg', '--config_file', help='the config file of the model',
        default='configs/infer.yaml')

    parser.add_argument(
        '-ut', '--url_txt',
        help='text file containing url of an image each line to be processed', default='')
    parser.add_argument(
        '-ul', '--url_list', nargs='+',
        help='list of urls of images to be processed', default=[])
    parser.add_argument(
        '-ir', '--image_dir',
        help='directory to load images to infer')
    parser.add_argument(
        '-min_size', '--min_test_size',
        help='the minimum size of the test images (default: 800)', type=int, default=800)
    parser.add_argument(
        '-th', '--confidence_threshold',
        help='the confidence threshold of the bounding boxes (default: 0.3)',
        type=float, default=0.6)
    parser.add_argument(
        '-t', '--target',
        help='the objects want to detect, support car and person', default='car'
    )
    parser.add_argument(
        '-vid', '--video_dir',
        help='Enter the name of the video file moved in demo folder', default='test_video'
    )		
    parser.add_argument(
        '-v', '--visualize', type=distutils.util.strtobool, default=False)
    parser.add_argument(
        '-vis_color', default='rainbow')


    args = parser.parse_args()
    return args


def main():
    """ main function """
    args = parse_args()
    video = False
    config_file = args.config_file
    assert config_file

    assert args.url_list or args.url_txt or args.image_dir or args.video_dir
    if len(args.url_list) > 0:
        url_list = args.url_list
    elif args.url_txt:
        url_list = list(np.loadtxt(args.url_txt, dtype=str))
    elif args.video_dir:
        video_dir = args.video_dir
        cap = cv2.VideoCapture(video_dir)
        video = True
    else:
        image_dir = args.image_dir
        url_list = [os.path.join(image_dir, item) for item in os.listdir(image_dir)]
    save_image = True if args.visualize else False

    target = args.target
    vis_color = args.vis_color

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=args.min_test_size,
        confidence_threshold=args.confidence_threshold,
    )
    if target == 'person':
        coco_demo.CATEGORIES = ["__background", "person"]

    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, now+'.json')


    record_dict = {'model': cfg.MODEL.WEIGHT,
                   'time': now,
                   'results': []}

    if(video):
        video_name = video_dir.split('/')[-1].split('.')[0]
        output_path = os.path.join(output_dir, video_name+'.json')
        skipped_frames_dir = output_dir +'/' + video_name +'_skipped_frames'
        if os.path.exists(skipped_frames_dir):
            shutil.rmtree(skipped_frames_dir)

        os.makedirs(skipped_frames_dir)
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        print(width,height)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter()
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out.open(os.path.join(output_dir, video_name + '.mp4'), fourcc, fps, (width, height), True)
        while(cap.isOpened()):
            ret, curr_frame = cap.read()
            curr_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if(ret):
                try:
                #if 2>1:
                    predictions = coco_demo.compute_prediction(curr_frame)
                    top_predictions = coco_demo.select_top_predictions(predictions)
                    scores = top_predictions.get_field("scores")
                    labels = top_predictions.get_field("labels")
                    boxes = predictions.bbox
                   #predictions.fields() - ' ['labels', 'scores', 'keypoints']'

                    keypoints = top_predictions.get_field("keypoints")
                    scores_keypoints = keypoints.get_field("logits")

                    #kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob)
                    kps = keypoints.keypoints

                    #replaces third column of KPS with confidence value for each keypoint
                    kps_cat = torch.cat((kps[:, :, 0:2], scores_keypoints[:, :, None]), dim=2).numpy()

                    infer_result = {'url': curr_frame_number,
                                    'boxes': [],
                                    'scores': [],
                                    'labels': [],
                                    'keypoints': []}
                    for box, score, label, keypts in zip(boxes, scores, labels, kps_cat):
                        boxpoints = [item for item in box.tolist()]
                        infer_result['boxes'].append(boxpoints)
                        infer_result['scores'].append(score.item())
                        infer_result['labels'].append(label.item())
                        infer_result['keypoints'].append(keypts.tolist())
                    record_dict['results'].append(infer_result)

                    # visualize the results
                    if save_image:
                        result = np.copy(curr_frame)
                        #result = coco_demo.overlay_boxes(result, top_predictions)
                        #result = coco_demo.overlay_class_names(result, top_predictions)
                        if cfg.MODEL.KEYPOINT_ON:
                            if target == 'person':
                                result = coco_demo.overlay_keypoints_graph(result, top_predictions, target='person')
                            if target == 'car':
                                result = coco_demo.overlay_keypoints_graph(result, top_predictions,vis_color , target='car')
                        out.write(result)
                        print('Processed frame ' + str(curr_frame_number))
                except:
                    print('Fail to infer for image {}. Skipped.'.format(str(curr_frame_number)))
                    cv2.imwrite(os.path.join(skipped_frames_dir, str(curr_frame_number)) + '.jpg', curr_frame)
                    continue
            elif not ret:
                print('Could not read frame', str(curr_frame_number))
                #cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_number + 1)
                break
                cap.release()
                out.release()

            else:
                break

    print(now)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(now)
    if(video):
        with open(output_path, 'w') as f:
            json.dump(record_dict, f)
            exit()




    for url in url_list:
        if not os.path.exists(url):
            print('Image {} does not exist!'.format(url))
            continue
        img = cv2.imread(url)

        #predictions = coco_demo.compute_prediction(img)
        #top_predictions = coco_demo.select_top_predictions(predictions)
        #print(top_predictions.get_field("keypoints").Keypoints[0])
        try:
        #if 2>1:
            predictions = coco_demo.compute_prediction(img)
            top_predictions = coco_demo.select_top_predictions(predictions)

            scores = top_predictions.get_field("scores")
            labels = top_predictions.get_field("labels")
            boxes = predictions.bbox

            keypoints = top_predictions.get_field("keypoints")
            scores_keypoints = keypoints.get_field("logits")

            #kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob)
            kps = keypoints.keypoints

            #replaces third column of KPS with confidence value for each keypoint
            kps_cat = torch.cat((kps[:, :, 0:2], scores_keypoints[:, :, None]), dim=2).numpy()

            infer_result = {'url': url,
                            'boxes': [],
                            'scores': [],
                            'labels': [],
                            'keypoints': []}
            for box, score, label, keypts in zip(boxes, scores, labels, kps_cat):
                boxpoints = [item for item in box.tolist()]
                infer_result['boxes'].append(boxpoints)
                infer_result['scores'].append(score.item())
                infer_result['labels'].append(label.item())
                infer_result['keypoints'].append(keypts.tolist())
            record_dict['results'].append(infer_result)
            # visualize the results
            if save_image:
                result = np.copy(img)
                #result = coco_demo.overlay_boxes(result, top_predictions)
                #result = coco_demo.overlay_class_names(result, top_predictions)
                if cfg.MODEL.KEYPOINT_ON:
                     if target == 'person':
                        result = coco_demo.overlay_keypoints_graph(result, top_predictions, target='person')
                     if target == 'car':
                        result = coco_demo.overlay_keypoints_graph(result, top_predictions,vis_color , target='car')
                cv2.imwrite(os.path.join(output_dir, url.split('/')[-1]), result)
                print(os.path.join(output_dir, url.split('/')[-1]))
        except:
            print('Fail to infer for image {}. Skipped.'.format(url))
            continue
    print(now)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(now)
   

    with open(output_path, 'w') as f:
        json.dump(record_dict, f)


if __name__ == '__main__':
    main()
