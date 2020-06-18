import numpy as np
import os
import math
import cv2
import argparse
import tensorflow as tf
from dataset_loader import Dataset
from net import RPN, RTN, FLN
from utils_np import *
from utils_tf import *
from config import *
import argparse

parser = argparse.ArgumentParser(description='Test all scenes in a dataset')
parser.add_argument('--dataset', type=str, help='dataset name: FIT, nuScenes, Waymo')
parser.add_argument('--output', help='write output images', action='store_true')
parser.add_argument('--noRPN', help='run the network without rpn and rtn', action='store_true')
args = parser.parse_args()
dataset_name = args.dataset
write_output_flag = args.output
no_rpn_flag = args.noRPN

output_folder = OUTPUT_FOLDER_FLN
if write_output_flag:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

width = DATASET_RESOLUTION[dataset_name][0]
height = DATASET_RESOLUTION[dataset_name][1]

if no_rpn_flag:
    model_path = FLN_noRPN_MODEL_PATH
else:
    model_path = FLN_MODEL_PATH
data_path = DATASET_PATH[dataset_name]

session = create_session()

# Define all placeholders for the input to the framework (i.e, the input to RPN, RTN and FLN)
x_static_segmentation_rpn = tf.placeholder(tf.float32, shape=(1, 3, RPN_RESOLUTION[1], RPN_RESOLUTION[0]))
x_static_segmentation = tf.placeholder(tf.float32, shape=(1, 3, FLN_RESOLUTION[1], FLN_RESOLUTION[0]))
x_objects = tf.placeholder(tf.float32, shape=(3, 1, 1, 6, 1))
x_imgs = tf.placeholder(tf.float32, shape=(3, 1, 3, FLN_RESOLUTION[1], FLN_RESOLUTION[0]))
x_semantics = tf.placeholder(tf.float32, shape=(3, 1, 3, FLN_RESOLUTION[1], FLN_RESOLUTION[0]))
x_egos = tf.placeholder(tf.float32, shape=(2, 1, 1, 7, 1))

# Build the network graph
output_rtn = []
if not no_rpn_flag:
    rpn_network = RPN(x_static_segmentation_rpn, x_objects[2])
    output_rpn = rpn_network.make_graph()
    resampled_output_rpn = tf_resample_hyps(output_rpn, float(RPN_RESOLUTION[0]/FLN_RESOLUTION[0]), float(RPN_RESOLUTION[1]/FLN_RESOLUTION[1]))
    rtn_network = RTN(x_static_segmentation, x_objects[2], x_imgs[2], x_egos, resampled_output_rpn)
    output_rtn = rtn_network.make_graph()

fln_network = FLN(x_imgs, x_semantics, x_egos, x_objects, output_rtn)
output_fln = fln_network.make_graph()

# Load the model snapshot
optimistic_restore(session, model_path)

# Load the input dataset
dataset = Dataset(data_path, dataset_name)

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
        static_segmentation_img = decode_semantic(testing_sequence.static_segmentations[0], width=FLN_RESOLUTION[0], height=FLN_RESOLUTION[1])
        objects_list = []
        imgs_list = []
        semantics_list = []
        for k in range(3):
            objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id, float(width/FLN_RESOLUTION[0]), float(height/FLN_RESOLUTION[1])))
            imgs_list.append(decode_img(testing_sequence.imgs[k], width=FLN_RESOLUTION[0], height=FLN_RESOLUTION[1]))
            semantics_list.append(decode_semantic(testing_sequence.segmentations[k], width=FLN_RESOLUTION[0], height=FLN_RESOLUTION[1]))
        objects = np.stack(objects_list, axis=0)
        imgs = np.stack(imgs_list, axis=0)
        semantics = np.stack(semantics_list, axis=0)
        egos = np.stack((decode_ego(testing_sequence.egos[0]), decode_ego(testing_sequence.egos[1])), axis=0)

        # means, sigmas have the format (tl_x, tl_y, br_x, br_y)
        # rtn_hyps, fln_hyps, gt have the format (center_x, center_y, w, h)
        means, sigmas, mixture_weights, fln_hyps, rtn_hyps = session.run(output_fln,
                                                                             feed_dict={x_static_segmentation_rpn: static_segmentation_img_rpn,
                                                                                        x_static_segmentation: static_segmentation_img,
                                                                                        x_objects: objects,
                                                                                        x_imgs: imgs,
                                                                                        x_semantics: semantics,
                                                                                        x_egos: egos
                                                                                        })

        gt_object = decode_obj_gt(testing_sequence.objects[-1], testing_sequence.id)
        fln_hyps_resampled = resample_hyps(fln_hyps, float(FLN_RESOLUTION[0] / width), float(FLN_RESOLUTION[1] / height))
        means_resampled = resample_hyps(means, float(FLN_RESOLUTION[0] / width), float(FLN_RESOLUTION[1] / height))
        sigmas_resampled = resample_hyps(sigmas, float(FLN_RESOLUTION[0] / width), float(FLN_RESOLUTION[1] / height))
        if write_output_flag:
            if not no_rpn_flag:
                rtn_hyps_resampled = resample_hyps(rtn_hyps, float(FLN_RESOLUTION[0] / width),
                                                   float(FLN_RESOLUTION[1] / height))
                drawn_img_rtn = draw_hyps(testing_sequence.imgs[-1], rtn_hyps_resampled, random_color=True)
                cv2.imwrite(os.path.join(result_scene_path, '%d-rtn.jpg' % i), drawn_img_rtn)
            draw_heatmap(testing_sequence.imgs[-1], means_resampled, sigmas, mixture_weights, width, height,
                         os.path.join(result_scene_path, '%d-fln.jpg' % i), gt=gt_object)
        nll = compute_nll(means_resampled, sigmas_resampled, mixture_weights, gt_object)
        fde = compute_oracle_FDE(fln_hyps_resampled, gt_object)
        iou = compute_oracle_IOU(fln_hyps_resampled, gt_object)
        print('NLL: %5.2f,\tFDE: %6.2f,\tIOU: %.2f' % (nll, fde, iou))
        nll_sum += nll
        fde_sum += fde
        iou_sum += iou
        counter += 1
print('--------------- AVERAGE METRICS ---------------')
print('NLL: %.2f, FDE: %.2f, IOU: %.2f, Number of samples: %d' %
      (nll_sum/counter, fde_sum/counter, iou_sum/counter, counter))