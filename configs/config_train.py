# config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.WORK = CN()
_C.WORK.PATH = "D:\\GitWork\\mnist\\"

_C.DATA = CN()
_C.DATA.PATH = "D:\\GitWork\\mnist\\data\\"
_C.DATA.RAW_PATH = "D:\\GitWork\\mnist\\data\\MNIST\\raw\\"
_C.DATA.PROCESSED_PATH = "D:\\GitWork\\mnist\\data\\MNIST\\processed\\"

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 100
_C.TRAIN.INPUT_SIZE = 784
_C.TRAIN.HIDDEN_SIZE1 = 128
_C.TRAIN.HIDDEN_SIZE2 = 64
_C.TRAIN.NUM_CLASSES = 10
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.LEARNING_RATE = 0.001
_C.TRAIN.MOMENTUM = 0.9


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()