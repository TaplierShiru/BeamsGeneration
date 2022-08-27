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


# +
CLASSIFICATION = 'class'
REGRESSION = 'regr'

TRAIN = 'train'
TEST = 'test'


# -

def load_data(
    path_exp_folder, path_to_data='expand_double_modes', 
    use_saved=False, size_hw=(336, 336), test_percent=0.2,
    data_type=CLASSIFICATION, filename_config='config_train_test'):
    """
    Load data with certain params

    Parameters
    ----------
    
    data_type : str
        CLASSIFICATION ('class') or REGRESSION ('regr')
    Returns
    -------
    Xtrain, Ytrain, Xtest, Ytest, pred2param, config_data

    """

    if use_saved:
        filename_config = filename_config.split('.')[0] # in case if type is present
        assert len(glob(f'{path_exp_folder}/{filename_config}.json')) == 1, 'where is no config file!' +\
               '\nbut use_saved was equal to False'
        with open(f'{path_exp_folder}/{filename_config}.json', 'r') as fp:
            config_data = json.load(fp)
    else:    
        config_data = {'train': [], 'test': []}
    Xtrain = []
    Xtest = []

    Ytrain = []
    Ytest = []
    pred2param = {}

    if not use_saved:
        print('Generate new data...')
        folders_path = sorted(glob(path_to_data + '/*'), key=lambda x: float(x.split('/')[-1]))
        print(len(folders_path))
        for i, single_folder in enumerate(folders_path):
            # Single folder - 1 class
            pred2param[i] = str(single_folder.split('/')[-1])
            if data_type == CLASSIFICATION:
                label = i
            elif data_type == REGRESSION:
                label = float(single_folder.split('/')[-1])
            else:
                raise TypeError('Wrong type for `data_type`')

            path_images = shuffle(glob(single_folder + '/*'))
            size = len(path_images)
            iterator = tqdm(enumerate(path_images))
            for j, single_img_path in iterator:
                readed_img = cv2.resize(cv2.imread(single_img_path)[..., ::-1], size_hw)
                if int(size * test_percent) < j:
                    # train
                    Xtrain.append(readed_img)
                    Ytrain.append(label)
                    config_data['train'] += [single_img_path]
                else:
                    # test
                    Xtest.append(readed_img)
                    Ytest.append(label)
                    config_data['test'] += [single_img_path]
            iterator.close()

        with open(f'{path_exp_folder}/{filename_config}.json', 'w') as fp:
            json.dump(config_data, fp)
    else:
        print("Read data from config file: ", filename_config)
        folders_path = sorted(glob(path_to_data + '/*'), key=lambda x: float(x.split('/')[-1]))
        print(f'Number of elements total: {len(folders_path)}')
        pred2param = dict([(i, num_a.split('/')[-1]) for i, num_a in enumerate(folders_path)])
        param2pred = dict([(num_a.split('/')[-1], i) for i, num_a in enumerate(folders_path)])
        # train
        iterator = tqdm(config_data['train'])
        for single_img_path in iterator:
            readed_img = cv2.resize(cv2.imread(os.path.join(path_to_data, single_img_path))[..., ::-1], size_hw)
            if data_type == CLASSIFICATION:
                label = param2pred[single_img_path.split('/')[-2]]
            elif data_type == REGRESSION:
                label = float(single_img_path.split('/')[-2])
            else:
                raise TypeError('Wrong type for `data_type`')

            Xtrain.append(readed_img)
            Ytrain.append(label)
        iterator.close()
        # test
        iterator = tqdm(config_data['test'])
        for single_img_path in iterator:
            readed_img = cv2.resize(cv2.imread(os.path.join(path_to_data, single_img_path))[..., ::-1], size_hw)
            if data_type == CLASSIFICATION:
                label = param2pred[single_img_path.split('/')[-2]]
            elif data_type == REGRESSION:
                label = float(single_img_path.split('/')[-2])
            else:
                raise TypeError('Wrong type for `data_type`')

            Xtest.append(readed_img)
            Ytest.append(label)
        iterator.close()

    print('train : ', len(Ytrain))
    print('test: ', len(Ytest))
    assert len(Xtrain) == len(Ytrain)
    assert len(Xtest) == len(Ytest)

    return Xtrain, Ytrain, Xtest, Ytest, pred2param, config_data


def load_data_minor(
    path_exp_folder, foldername2class, path_to_data='expand_double_modes', 
    size_hw=(336, 336), test_percent=0.2, use_saved=False,
    data_type=CLASSIFICATION, filename_config='config_train_test', index_shift=1):
    """
    Load data with certain params

    Parameters
    ----------
    
    data_type : str
        CLASSIFICATION ('class') or REGRESSION ('regr')
    Returns
    -------
    Xtrain, Ytrain, Xtest, Ytest, pred2param, config_data

    """
    if use_saved:
        filename_config = filename_config.split('.')[0] # in case if type is present
        assert len(glob(f'{path_exp_folder}/{filename_config}.json')) == 1, 'where is no config file!' +\
               '\nbut use_saved was equal to False'
        with open(f'{path_exp_folder}/{filename_config}.json', 'r') as fp:
            config_data = json.load(fp)
    else:    
        config_data = {'train': [], 'test': []}

    class2folders_path = dict([
        (str(float(s_f.split('/')[-1])), s_f)
        for s_f in sorted(glob(path_to_data + '/*'), key=lambda x: float(x.split('/')[-1]))
    ])
    Xtrain = []
    Xtest = []

    Ytrain = []
    Ytest = []
    pred2param = {}

    if not use_saved:
        print(len(foldername2class))
        for i, single_folder in enumerate(foldername2class):
            # Single folder - 1 class
            pred2param[i] = foldername2class[single_folder]
            if data_type == CLASSIFICATION:
                label = i
            elif data_type == REGRESSION:
                label = float(single_folder)
            else:
                raise TypeError('Wrong type for `data_type`')

            path_images = shuffle(glob(class2folders_path[single_folder] + '/*'))
            size = len(path_images)
            iterator = tqdm(enumerate(path_images))
            for j, single_img_path in iterator:
                readed_img = cv2.resize(cv2.imread(single_img_path)[..., ::-1], size_hw)
                if int(size * test_percent) < j:
                    # train
                    Xtrain.append(readed_img)
                    Ytrain.append(label)
                    config_data['train'] += [single_img_path]
                else:
                    # test
                    Xtest.append(readed_img)
                    Ytest.append(label)
                    config_data['test'] += [single_img_path]
            iterator.close()

        with open(f'{path_exp_folder}/{filename_config}.json', 'w') as fp:
            json.dump(config_data, fp)
    else:
        pred2param = dict([(str(num_a), i) for i, num_a in enumerate(foldername2class.items())])
        param2pred = foldername2class
        # train
        iterator = tqdm(config_data['train'])
        for single_img_path in iterator:
            readed_img = cv2.resize(cv2.imread(single_img_path)[..., ::-1], size_hw)
            if data_type == CLASSIFICATION:
                label = param2pred[single_img_path.split('/')[index_shift]]
            elif data_type == REGRESSION:
                label = float(single_img_path.split('/')[index_shift])
            else:
                raise TypeError('Wrong type for `data_type`')

            Xtrain.append(readed_img)
            Ytrain.append(label)
        iterator.close()
        # test
        iterator = tqdm(config_data['test'])
        for single_img_path in iterator:
            readed_img = cv2.resize(cv2.imread(single_img_path)[..., ::-1], size_hw)
            if data_type == CLASSIFICATION:
                label = param2pred[single_img_path.split('/')[index_shift]]
            elif data_type == REGRESSION:
                label = float(single_img_path.split('/')[index_shift])
            else:
                raise TypeError('Wrong type for `data_type`')

            Xtest.append(readed_img)
            Ytest.append(label)
        iterator.close()

    print('train : ', len(Ytrain))
    print('test: ', len(Ytest))


    return Xtrain, Ytrain, Xtest, Ytest, pred2param, config_data

