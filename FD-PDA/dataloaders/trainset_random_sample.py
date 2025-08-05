import os
from cProfile import label

import cv2
import torch
import random
import numpy as np
from glob import glob

import json

from openpyxl.styles.builtins import output

ratio = 1

if __name__ == '__main__':

    base_dir = '../../data/CAMUS'
    json_path = os.path.join(base_dir, "frame_split.json")
    print(os.path.abspath(json_path))
    with open(json_path, 'r') as f1:
        split_json = json.load(f1) # number idx


    all_labeled_idxs = np.array(split_json['train_labeled_idx'])
    ori_unlabeled_idxs = np.array(split_json['train_unlabeled_idx'])

    labeled_num = int (len(all_labeled_idxs)//4 // ratio)
    all_patients = np.arange(0,len(all_labeled_idxs), 4)
    labeled = np.array([], dtype=np.int8)

    selected_labeled = np.random.choice(all_patients, labeled_num, replace=False)

    for i in selected_labeled:
        labeled = np.append(labeled, np.arange(i,i+4))

    unselected = np.setdiff1d(np.arange(0,1600), labeled)
    labeled = np.sort(labeled)

    labeled_idxs = all_labeled_idxs[labeled]
    unlabeled_idxs = np.sort( np.append(all_labeled_idxs[unselected], ori_unlabeled_idxs) )

    split_json['train_labeled_idx'] = labeled_idxs.tolist()
    split_json['train_unlabeled_idx'] = unlabeled_idxs.tolist()


    output_path = os.path.join(base_dir, "frame_split_{}.json".format(labeled_num) )
    with open(output_path, 'w') as json_file:
        json.dump(split_json, json_file, indent=4)

    with open(output_path, 'r') as f1:
        split_json1 = json.load(f1)
    print (len(split_json1['train_labeled_idx'] ))