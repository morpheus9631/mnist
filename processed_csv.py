from __future__ import print_function, division  

import os, sys
import argparse
import codecs
import gzip
import numpy as np
import shutil
import time
import torch
from configs.config_train import  get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(description='Ants and Bees by PyTorch')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()
   

def main():
    args = parse_args()
    print(args)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print('\n', cfg)

    RawPath = cfg.DATA.RAW_PATH
    ProcessedPath = cfg.DATA.PROCESSED_PATH

    if not os.path.exists(ProcessedPath):
        os.makedirs(ProcessedPath)

    resources = [{ 
            "image": "train-images-idx3-ubyte", 
            "label": "train-labels-idx1-ubyte",
            "count": 60000,
            "outFname": "mnist_train.csv"
        }, {
            "image": "t10k-images-idx3-ubyte",  
            "label": "t10k-labels-idx1-ubyte",
            "count": 10000,
            "outFname": "mnist_test.csv"
        }]

    for r in resources:
        proc_name = r['outFname']
        print("\n'{}' processing...".format(proc_name))
        
        img_path = os.path.join(RawPath, r['image'])
        lbl_path = os.path.join(RawPath, r['label'])
        out_path = os.path.join(ProcessedPath, r['outFname'])
        cnt = r['count']
        convert(img_path, lbl_path, out_path, cnt)
        
        print("'{}' processed.".format(proc_name))

    print('\ndone')

    return (0)

        
if __name__=='__main__':
    main()