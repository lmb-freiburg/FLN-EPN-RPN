import numpy as np
import os
import math
import cv2
import argparse
import tensorflow as tf
from dataset_loader import Dataset
from net import RPN, RTN, EPN
from utils_np import *
from utils_tf import *
from config import *
import argparse

parser = argparse.ArgumentParser(description='Test all scenes in a dataset')
parser.add_argument('--output', help='write output images', action='store_true')
parser.add_argument('--noRPN', help='run the network without rpn and rtn', action='store_true')
args = parser.parse_args()
dataset_name = 'nuScenes' # only nuScenes is supported now
write_output_flag = args.output
no_rpn_flag = args.noRPN

output_folder = OUTPUT_FOLDER_EPN

if write_output_flag:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

width = DATASET_RESOLUTION[dataset_name][0]
height = DATASET_RESOLUTION[dataset_name][1]

if no_rpn_flag:
    model_path = EPN_noRPN_MODEL_PATH
else:
    model_path = EPN_MODEL_PATH

data_path = DATASET_PATH[dataset_name]

session = create_session()

# Define all placeholders for the input to the framework (i.e, the input to RPN, RTN and FLN)
x_static_segmentation_rpn = tf.placeholder(tf.float32, shape=(1, 3, RPN_RESOLUTION[1], RPN_RESOLUTION[0]))
x_static_segmentation = tf.placeholder(tf.float32, shape=(1, 3, EPN_RESOLUTION[1], EPN_RESOLUTION[0]))
x_object = tf.placeholder(tf.float32, shape=(1, 1, 6, 1))
x_img = tf.placeholder(tf.float32, shape=(1, 3, EPN_RESOLUTION[1], EPN_RESOLUTION[0]))
x_semantic = tf.placeholder(tf.float32, shape=(1, 3, EPN_RESOLUTION[1], EPN_RESOLUTION[0]))
x_egos = tf.placeholder(tf.float32, shape=(2, 1, 1, 7, 1))

# Build the network graph
output_rtn = []
if not no_rpn_flag:
    rpn_network = RPN(x_static_segmentation_rpn, x_object)
    output_rpn = rpn_network.make_graph()
    resampled_output_rpn = tf_resample_hyps(output_rpn, float(RPN_RESOLUTION[0]/EPN_RESOLUTION[0]), float(RPN_RESOLUTION[1]/EPN_RESOLUTION[1]))
    rtn_network = RTN(x_static_segmentation, x_object, x_img, x_egos, resampled_output_rpn)
    output_rtn = rtn_network.make_graph()
epn_network = EPN(x_img, x_semantic, x_egos, x_object, output_rtn)
output_epn = epn_network.make_graph()

# Load the model snapshot
optimistic_restore(session, model_path)

# Load the input dataset
dataset = Dataset(data_path, dataset_name, type='EPN')

nll_sum = 0
fde_sum = 0
iou_sum = 0
counter = 0
# Run the test for each sequence for each scene
for scene_index in range(len(dataset.scenes)):
    scene = dataset.scenes[scene_index]
    scene_name = scene.scene_path.split('/')[-1]
    print('---------------- Scene %s ---------------------' % scene_name)
    if write_output_flag:
        result_scene_path = os.path.join(output_folder, dataset_name, scene_name)
        os.makedirs(result_scene_path, exist_ok=True)
    for i in range(len(scene.sequences)):
        testing_sequence = scene.sequences[i]
        static_segmentation_img_rpn = decode_semantic(testing_sequence.static_segmentations[0], width=RPN_RESOLUTION[0], height=RPN_RESOLUTION[1])
        static_segmentation_img = decode_semantic(testing_sequence.static_segmentations[0], width=EPN_RESOLUTION[0], height=EPN_RESOLUTION[1])
        object = decode_obj(testing_sequence.objects[-1], testing_sequence.id, float(width/EPN_RESOLUTION[0]), float(height/EPN_RESOLUTION[1]))
        img = decode_img(testing_sequence.imgs[0], width=EPN_RESOLUTION[0], height=EPN_RESOLUTION[1])
        semantic = decode_semantic(testing_sequence.segmentations[0], width=EPN_RESOLUTION[0], height=EPN_RESOLUTION[1])
        egos = np.stack((decode_ego(testing_sequence.egos[0]), decode_ego(testing_sequence.egos[1])), axis=0)

        # means, sigmas have the format (tl_x, tl_y, br_x, br_y)
        # rtn_hyps, fln_hyps, gt have the format (center_x, center_y, w, h)
        means, sigmas, mixture_weights, epn_hyps, rtn_hyps = session.run(output_epn,
                                                                         feed_dict={x_static_segmentation_rpn: static_segmentation_img_rpn,
                                                                                    x_static_segmentation: static_segmentation_img,
                                                                                    x_object: object,
                                                                                    x_img: img,
                                                                                    x_semantic: semantic,
                                                                                    x_egos: egos
                                                                                    })

        gt_object = decode_obj_gt(testing_sequence.objects[-1], testing_sequence.id)
        epn_hyps_resampled = resample_hyps(epn_hyps, float(EPN_RESOLUTION[0] / width), float(EPN_RESOLUTION[1] / height))
        means_resampled = resample_hyps(means, float(EPN_RESOLUTION[0] / width), float(EPN_RESOLUTION[1] / height))
        sigmas_resampled = resample_hyps(sigmas, float(EPN_RESOLUTION[0] / width), float(EPN_RESOLUTION[1] / height))
        if write_output_flag:
            if not no_rpn_flag:
                rtn_hyps_resampled = resample_hyps(rtn_hyps, float(EPN_RESOLUTION[0] / width),
                                                   float(EPN_RESOLUTION[1] / height))
                drawn_img_rtn = draw_hyps(testing_sequence.imgs[-1], rtn_hyps_resampled, random_color=True)
                cv2.imwrite(os.path.join(result_scene_path, '%d-rtn.jpg' % i), drawn_img_rtn)
            draw_heatmap(testing_sequence.imgs[-1], means_resampled, sigmas, mixture_weights, width, height,
                         os.path.join(result_scene_path, '%d-epn.jpg' % i), gt=gt_object)
        nll = compute_nll(means_resampled, sigmas_resampled, mixture_weights, gt_object)
        fde = compute_oracle_FDE(epn_hyps_resampled, gt_object)
        iou = compute_oracle_IOU(epn_hyps_resampled, gt_object)
        print('NLL: %5.2f,\tFDE: %6.2f,\tIOU: %.2f' % (nll, fde, iou))
        nll_sum += nll
        fde_sum += fde
        iou_sum += iou
        counter += 1
print('--------------- AVERAGE METRICS ---------------')
print('NLL: %.2f, FDE: %.2f, IOU: %.2f, Number of samples: %d' %
      (nll_sum/counter, fde_sum/counter, iou_sum/counter, counter))