import numpy as np
import os
from config import *
from utils_np import *

class Dataset():
    def __init__(self, path, dataset_name, type='FLN'):
        self.path = path
        self.dataset_name=dataset_name
        self.type = type
        self.scenes = []
        self.load_scenes()

    def load_scenes(self):
        scenes_names = sorted(os.listdir(self.path))
        for scene_name in scenes_names:
            if self.type == 'FLN':
                if os.path.exists(os.path.join(self.path, scene_name, 'scene.txt')):
                    self.scenes.append(Scene(os.path.join(self.path, scene_name), self.dataset_name))
            elif self.type == 'EPN':
                if os.path.exists(os.path.join(self.path, scene_name, 'scene_epn.txt')):
                    self.scenes.append(Scene_EPN(os.path.join(self.path, scene_name), self.dataset_name))
            else:
                print('Unknown Dataset Type: it should be FLN or EPN')


class Scene():
    def __init__(self, scene_path, dataset_name):
        self.scene_path = scene_path
        self.dataset_name=dataset_name
        self.img_ext = DATASET_IMG_EXTENSION[dataset_name]
        self.semantic_ext = DATASET_SEMANTIC_EXTENSION[dataset_name]
        self.semantic_static_ext = DATASET_SEMANTIC_STATIC_EXTENSION[dataset_name]
        self.sequences = []
        self.load_sequences()

    def parse_string(self, c):
        id, ss = c.split(' ')
        img_0, img_1, img_2, img_f = ss.split(',')
        return int(id), str(img_0).strip(), str(img_1).strip(), str(img_2).strip(), str(img_f).strip()

    def load_sequences(self):
        with open(os.path.join(self.scene_path, 'scene.txt')) as f:
            content = f.readlines()
            for c in content:
                id, img_0, img_1, img_2, img_f = self.parse_string(c)
                img_0_path = os.path.join(self.scene_path, 'imgs', '%s%s' % (img_0, self.img_ext))
                img_1_path = os.path.join(self.scene_path, 'imgs', '%s%s' % (img_1, self.img_ext))
                img_2_path = os.path.join(self.scene_path, 'imgs', '%s%s' % (img_2, self.img_ext))
                img_f_path = os.path.join(self.scene_path, 'imgs', '%s%s' % (img_f, self.img_ext))

                obj_0_path = os.path.join(self.scene_path, 'floats', '%s.float3' % img_0)
                obj_1_path = os.path.join(self.scene_path, 'floats', '%s.float3' % img_1)
                obj_2_path = os.path.join(self.scene_path, 'floats', '%s.float3' % img_2)
                obj_f_path = os.path.join(self.scene_path, 'floats', '%s.float3' % img_f)

                seg_0_path = os.path.join(self.scene_path, 'semantics', '%s%s' % (img_0, self.semantic_ext))
                seg_1_path = os.path.join(self.scene_path, 'semantics', '%s%s' % (img_1, self.semantic_ext))
                seg_2_path = os.path.join(self.scene_path, 'semantics', '%s%s' % (img_2, self.semantic_ext))
                static_seg_2_path = os.path.join(self.scene_path, 'semantics', '%s%s' % (img_2, self.semantic_static_ext))

                ego_0_path = os.path.join(self.scene_path, 'ego', '%s.float3' % img_2)
                ego_f_path = os.path.join(self.scene_path, 'ego', '%s.float3' % img_f)

                self.sequences.append(Sequence(id, [img_0_path, img_1_path, img_2_path, img_f_path],
                                               [seg_0_path, seg_1_path, seg_2_path], [static_seg_2_path],
                                               [obj_0_path, obj_1_path, obj_2_path, obj_f_path], [ego_0_path, ego_f_path]))

class Scene_EPN():
    def __init__(self, scene_path, dataset_name):
        self.scene_path = scene_path
        self.dataset_name=dataset_name
        self.img_ext = DATASET_IMG_EXTENSION[dataset_name]
        self.semantic_ext = DATASET_SEMANTIC_EXTENSION[dataset_name]
        self.semantic_static_ext = DATASET_SEMANTIC_STATIC_EXTENSION[dataset_name]
        self.sequences = []
        self.load_sequences()

    def parse_string(self, c):
        id, ss = c.split(' ')
        img_0, img_f = ss.split(',')
        return int(id), str(img_0).strip(), str(img_f).strip()

    def load_sequences(self):
        with open(os.path.join(self.scene_path, 'scene_epn.txt')) as f:
            content = f.readlines()
            for c in content:
                id, img_0, img_f = self.parse_string(c)
                img_0_path = os.path.join(self.scene_path, 'imgs', '%s%s' % (img_0, self.img_ext))
                img_f_path = os.path.join(self.scene_path, 'imgs', '%s%s' % (img_f, self.img_ext))

                obj_0_path = os.path.join(self.scene_path, 'floats_epn', '%s.float3' % img_0)
                obj_f_path = os.path.join(self.scene_path, 'floats_epn', '%s.float3' % img_f)

                seg_0_path = os.path.join(self.scene_path, 'semantics', '%s%s' % (img_0, self.semantic_ext))
                static_seg_0_path = os.path.join(self.scene_path, 'semantics', '%s%s' % (img_0, self.semantic_static_ext))

                ego_0_path = os.path.join(self.scene_path, 'ego', '%s.float3' % img_0)
                ego_f_path = os.path.join(self.scene_path, 'ego', '%s.float3' % img_f)

                self.sequences.append(Sequence(id, [img_0_path, img_f_path],
                                               [seg_0_path], [static_seg_0_path],
                                               [obj_0_path, obj_f_path], [ego_0_path, ego_f_path]))

class Sequence():
    def __init__(self, id, imgs, segmentations, static_segmentations, objects, egos):
        self.id = id
        self.imgs = imgs
        self.segmentations = segmentations
        self.static_segmentations = static_segmentations
        self.objects = objects
        self.egos = egos