import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
import random
from PIL import Image
import seaborn as sns
import PIL
import json

from .process import img_RGB2LAB_preprocess_np

from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import shuffle


WRONG_PREDICTIONS = 100000.0


def save_conf_matrix(Ytest, predictions, path_save, save_np=True, num_classes=21):
    c_mat = confusion_matrix(
        Ytest[:len(predictions)].reshape(-1), predictions.reshape(-1), 
        labels=range(num_classes), normalize='true'
    ).astype(np.float32) 
    c_mat = np.round(c_mat, 2).astype(np.float32, copy=False)
    
    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(c_mat, annot=True)
    fig.savefig(path_save + '.png')
    plt.close('all')
    if save_np:
        np.save(path_save + '.npy', c_mat)


def _rotation_range():
    return range(-90, 90, 5)


def error_angle(path_save, config_data, Ytest, predictions, class_find=list(range(21)), save_np=True):
    y_z = Ytest[:len(predictions)].copy()
    p_z = predictions.copy()

    # dict - dict that store error of every angle, 
    # Each elem is array of size 2, 
    # 0-th - number of wrong preds, 1-th - number of all images
    angle_count = dict([(str(i), 0) for i in _rotation_range()])
    pred_data_angle, true_data_angle = [], []
    test_y, test_p = [], []

    for i in range(len(predictions)):
        # Skip class certain classes, (in order to collect data for certain classes)
        if y_z[i] not in class_find:
            continue
        # 90, 95, -55 and etc...
        angle_info = config_data['test'][i].split('/')[-1].split('.')[0].split('_')[-1]
        # 0.0, 0.2 and etc...
        #class_name_z = config_data['test'][i].split('/')[1]
        
        if p_z[i] != y_z[i]:
            pred_data_angle.append(float(angle_info))
            true_data_angle.append(float(angle_info))
        else:
            pred_data_angle.append(WRONG_PREDICTIONS)
            true_data_angle.append(float(angle_info))
        angle_count[angle_info] += 1
        test_y.append(y_z[i])
        test_p.append(p_z[i])

    c_mat = confusion_matrix(true_data_angle, pred_data_angle, labels=_rotation_range()).astype(np.float32) 
    for i, z in enumerate(_rotation_range()):
        c_mat[i, i] /= float(angle_count[str(z)])
    fig, ax = plt.subplots(figsize=(26, 26), facecolor='white')
    sns.heatmap(
        c_mat, annot=True, 
        xticklabels=list(_rotation_range()), yticklabels=list(_rotation_range())
    )
    fig.savefig(path_save + '.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')
    if save_np:
        np.save(path_save + '.npy', c_mat)

