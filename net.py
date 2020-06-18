import os
import math
import argparse
import tensorflow as tf
from resnet import ResNetArchitecture
from utils_tf import *

class RPN():
    def __init__(self, static, object): # pass the last observed static semantic segmentation and the object of interest
        # self.static = tf_resample_semantic(static, width=448, height=256)
        # self.static = tf_resample(self.static, width=448, height=320)
        self.static = static
        self.object = object # (1, 1, 6, 1)

    def disassembling(self, data):  # input has shape (1, 80, 1, 1)
        hyps = tf.split(data, [4 for i in range(20)], 1)  # set of (1, 4, 1, 1)
        return hyps

    def prepare_input(self):
        target_id = self.object[0, 0, 5, 0] # (55 for car, 19 for pedestrian)
        target_blob = tf.fill((1, 1, self.static.shape[2], self.static.shape[3]), target_id)
        input = tf.concat([self.static, target_blob], axis=1)
        return input

    def make_graph(self):
        arch = ResNetArchitecture(num_classes=4*20, avg_pool=False, batch_norm_flag=False,
                                  resnet_version=1, reuse=False)
        input = self.prepare_input()
        output = arch.make_graph(input, training=False)
        out_hyps = self.disassembling(output)
        return out_hyps

class RTN():
    def __init__(self, static, object, img, egos, rpn_hyps):
        #self.static = tf_resample_semantic(static, width=448, height=256)
        self.static = static
        self.object = object # (1, 1, 6, 1)
        #self.img = tf_resample_img(img, width=448, height=256)
        self.img = img
        self.egos = egos # (2,1,1,7,1) for 2 egos (first is for the last observed image and second is for the future image)
        self.rpn_hyps = rpn_hyps

    def disassembling(self, data):  # input has shape (1, 80, 1, 1)
        hyps = tf.split(data, [4 for i in range(20)], 1)  # set of (1, 4, 1, 1)
        return hyps

    def prepare_input(self):
        [n, c, h, w] = self.img.get_shape().as_list()
        input_list = []
        input_list.append(self.img)
        input_list.append(self.static)
        multiply = tf.constant([1, 1, h, w])
        for hyp in self.rpn_hyps:  # n,4,1,1
            tiled = tf.tile(hyp, multiply)
            input_list.append(tf.stop_gradient(tiled))
        input = tf.concat(input_list, axis=1)
        return input

    def make_graph(self):
        with tf.variable_scope("transfer"):
            arch = ResNetArchitecture(num_classes=4*20, avg_pool=False, batch_norm_flag=False,
                                      resnet_version=1, reuse=False, shallow_reduction=True)
            input = self.prepare_input()
            relative_pose = (self.egos[1] - self.egos[0])[:, 0, :, 0]  # 1,1,7,1 >> 1,7
            pose_feat = tf.layers.dense(inputs=relative_pose, units=512, activation=tf.nn.relu)
            output = arch.make_graph(input, training=False, additional_features=pose_feat)
        out_hyps = self.disassembling(output)
        for i in range(len(self.rpn_hyps)):
                out_hyps[i] = self.rpn_hyps[i] + out_hyps[i]
        return out_hyps

class FLN():
    def __init__(self, imgs, semantics, egos, objects, rtn_hyps):
        self.imgs = imgs
        self.semantics = semantics
        self.egos = egos
        self.objects = objects
        self.rtn_hyps = rtn_hyps

    def disassembling(self, data):  # input has shape (1, 80, 1, 1)
        hyps = tf.split(data, [4 for i in range(20)], 1)  # set of (1, 4, 1, 1)
        return hyps

    def prepare_input(self):
        [n, c, h, w] = self.imgs[0].get_shape().as_list()
        input_list = []
        all_masks = tf.zeros((n, 1, h, w))
        for i in range(0, 3):
            object = self.objects[i]
            mask = tf_get_mask(object[0, 0, :, 0], w, h, fill_value=object[0, 0, 5, 0])
            all_masks = all_masks + mask
            input_list.append(self.imgs[i])
            input_list.append(self.semantics[i])
            input_list.append(mask)
        if len(self.rtn_hyps) > 0:
            imposed_image = tf_impose_hyps(self.rtn_hyps, w, h)
            input_list.append(imposed_image)
            multiply = tf.constant([1, 1, h, w])
            for hyp in self.rtn_hyps:  # n,4,1,1
                tiled = tf.tile(hyp, multiply)
                input_list.append(tiled)
        input = tf.concat(input_list, axis=1)
        return input

    def make_graph(self):
        with tf.variable_scope("prediction"):
            arch = ResNetArchitecture(num_classes=4 * 20, avg_pool=False, batch_norm_flag=False,
                                      resnet_version=1, reuse=False, shallow_reduction=True)
            input = self.prepare_input()
            relative_pose = (self.egos[1] - self.egos[0])[:, 0, :, 0]  # 1,1,7,1 >> 1,7
            pose_feat = tf.layers.dense(inputs=relative_pose, units=512, activation=tf.nn.relu)
            output = arch.make_graph(input, training=False, additional_features=pose_feat)
        out_hyps = self.disassembling(output)

        # fitting
        with tf.variable_scope("fitting"):
            intermediate = tf.tanh(tf_full_conn(output, name='predict_fc0', num_output=500))
            intermediate_drop = intermediate
            predicted = tf_full_conn(intermediate_drop, name='predict_fc1', num_output=20 * 4)
        out_soft_assignments = self.disassembling(predicted)
        means, bounded_log_sigmas, mixture_weights = tf_assemble_gmm_parameters_samples(samples_means=out_hyps, assignments=out_soft_assignments)
        sigmas = [tf.exp(x) for x in bounded_log_sigmas]

        return means, sigmas, mixture_weights, out_hyps, self.rtn_hyps


class EPN():
    def __init__(self, img, semantic, egos, object, rtn_hyps):
        self.img = img
        self.semantic = semantic
        self.egos = egos
        self.object = object
        self.rtn_hyps = rtn_hyps

    def disassembling(self, data):  # input has shape (1, 80, 1, 1)
        hyps = tf.split(data, [4 for i in range(20)], 1)  # set of (1, 4, 1, 1)
        return hyps

    def disassembling_assignments(self, data):  # input has shape (1, 80, 1, 1)
        hyps = tf.split(data, [8 for i in range(20)], 1)  # set of (1, 4, 1, 1)
        return hyps

    def prepare_input(self):
        [n, c, h, w] = self.img.get_shape().as_list()
        input_list = [self.img]
        target_id = self.object[0, 0, 5, 0]  # (55 for car, 19 for pedestrian)
        target_blob = tf.fill((1, 1, self.img.shape[2], self.img.shape[3]), target_id)
        input_list.append(target_blob)
        input_list.append(self.semantic)
        if len(self.rtn_hyps) > 0:
            imposed_image = tf_impose_hyps(self.rtn_hyps, w, h)
            input_list.append(imposed_image)
            multiply = tf.constant([1, 1, h, w])
            for hyp in self.rtn_hyps:  # n,4,1,1
                tiled = tf.tile(hyp, multiply)
                input_list.append(tiled)
        input = tf.concat(input_list, axis=1)
        return input

    def make_graph(self):
        with tf.variable_scope("anticipation"):
            arch = ResNetArchitecture(num_classes=4 * 20, avg_pool=False, batch_norm_flag=False,
                                      resnet_version=1, reuse=False, shallow_reduction=True)
            input = self.prepare_input()
            relative_pose = (self.egos[1] - self.egos[0])[:, 0, :, 0]  # 1,1,7,1 >> 1,7
            pose_feat = tf.layers.dense(inputs=relative_pose, units=512, activation=tf.nn.relu)
            output = arch.make_graph(input, training=False, additional_features=pose_feat)
        out_hyps = self.disassembling(output)

        # fitting
        with tf.variable_scope("fitting"):
            intermediate = tf.tanh(tf_full_conn(output, name='predict_fc0', num_output=500))
            intermediate_drop = intermediate
            predicted = tf_full_conn(intermediate_drop, name='predict_fc1', num_output=20 * 8)
        out_soft_assignments = self.disassembling_assignments(predicted)
        means, bounded_log_sigmas, mixture_weights = tf_assemble_gmm_parameters_samples(samples_means=out_hyps, assignments=out_soft_assignments)
        sigmas = [tf.exp(x) for x in bounded_log_sigmas]

        return means, sigmas, mixture_weights, out_hyps, self.rtn_hyps