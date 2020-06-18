
DATASET_PATH = {'FIT': 'datasets/FIT',
                'waymo': 'datasets/waymo',
                'nuScenes': 'datasets/nuScenes'}

DATASET_RESOLUTION = {'FIT': (576, 256),
                      'waymo': (512, 320),
                      'nuScenes': (512, 320)}

DATASET_IMG_EXTENSION = {'FIT': '.jpg',
                      'waymo': '.png',
                      'nuScenes': '.jpg'}

DATASET_SEMANTIC_EXTENSION = {'FIT': '.jpg',
                              'waymo': '-semantic.png',
                              'nuScenes': '-semantic.png'}

DATASET_SEMANTIC_STATIC_EXTENSION = {'FIT': '-static.png',
                                     'waymo': '-semantic-static.png',
                                     'nuScenes': '-semantic-static.png'}

FLN_MODEL_PATH = 'models/FLN/snapshot-200000'
EPN_MODEL_PATH = 'models/EPN/snapshot-200000'
FLN_noRPN_MODEL_PATH = 'models/FLN-no-RPN/snapshot-200000'
EPN_noRPN_MODEL_PATH = 'models/EPN-no-RPN/snapshot-200000'

RPN_RESOLUTION = (448, 320)
FLN_RESOLUTION = (448, 256)
EPN_RESOLUTION = (448, 256)

OUTPUT_FOLDER_FLN = 'result_FLN'
OUTPUT_FOLDER_EPN = 'result_EPN'