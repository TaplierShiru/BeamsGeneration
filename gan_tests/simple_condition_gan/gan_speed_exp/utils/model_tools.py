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

from process import img_RGB2LAB_preprocess_np
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import shuffle
from makizoo.backbones.resnetv1 import ResNet18
from makizoo.backbones.mobilenetv2 import MobileNetV2_1_4, MobileNetV2_1_0, MobileNetV2_0_75
from makiflow.layers import *
from makiflow.models.classificator import Classificator, CETrainer
from makiflow.generators.classification import cycle_generator
from makiflow.models.regressor import Regressor, AbsTrainer, MseTrainer


coffe_norm = lambda x: (x - 128.0) / 128.0
NUM_CLASSES = None
NUM_REGRESS_OUT = None


def get_regression_model_peak_from_pool(
    size_hw=(336, 336), 
    batch_size=64, 
    size_dataset=35000, 
    num_out=1, 
    lr=5e-3, 
    indx_model=0):
    """
    Create model

    Parameters
    ----------

    Returns
    -------
    model, trainer, opt, global_step, sess

    """
    if indx_model < 0 or indx_model >= 5:
        raise ValueError("wrong `indx_model` !!")

    global NUM_REGRESS_OUT
    inp, x = ResNet18([batch_size, size_hw[0], size_hw[1], 3])
    # best were: 3 (lr=8e-3), 0 (lr=2e-2)
    x = GlobalAvgPoolLayer('flat')(x)
    if indx_model == 0:
        # Head1
        x = DropoutLayer(name='drop_final_1', p_keep=0.5)(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=num_out, 
            use_bias=False, activation=None, name='regression_head_1'
        )(x)
    elif indx_model == 1:
        x = ActivationLayer(name='regress_act_0')(x)
        x = BatchNormLayer(D=x.get_shape()[-1], name='regress_bn_0')(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=x.get_shape()[-1], 
            use_bias=False, activation=None, name='regression_head_0'
        )(x)
        x = DropoutLayer(name='drop_final_1', p_keep=0.5)(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=num_out, 
            use_bias=False, activation=None, name='regression_head_1'
        )(x)
    elif indx_model == 2:
        x = DropoutLayer(name='drop_final_0', p_keep=0.7)(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=x.get_shape()[-1], 
            use_bias=False, activation=tf.nn.relu, name='regression_head_0'
        )(x)
        x = DropoutLayer(name='drop_final_1', p_keep=0.5)(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=num_out, 
            use_bias=False, activation=None, name='regression_head_1'
        )(x)
    elif indx_model == 3:
        x = ActivationLayer(name='regress_act_0')(x)
        x = BatchNormLayer(D=x.get_shape()[-1], name='regress_bn_0')(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=x.get_shape()[-1], 
            use_bias=False, activation=None, name='regression_head_0'
        )(x)
        x = ActivationLayer(name='regress_act_1')(x)
        x = BatchNormLayer(D=x.get_shape()[-1], name='regress_bn_1')(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=num_out, 
            use_bias=False, activation=None, name='regression_head_1'
        )(x)
    elif indx_model == 4:
        x = DropoutLayer(name='drop_final_0', p_keep=0.7)(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=x.get_shape()[-1], 
            use_bias=False, activation=None, name='regression_head_0'
        )(x)
        x = DropoutLayer(name='drop_final_1', p_keep=0.7)(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=x.get_shape()[-1], 
            use_bias=False, activation=None, name='regression_head_1'
        )(x)
        x = DropoutLayer(name='drop_final_2', p_keep=0.5)(x)
        x = DenseLayer(
            in_d=x.get_shape()[-1], out_d=num_out, 
            use_bias=False, activation=None, name='regression_head_2'
        )(x)

    """
    # best
    # Head1
    x = DropoutLayer(name='drop_final_1', p_keep=0.5)(x)
    x = DenseLayer(
        in_d=x.get_shape()[-1], out_d=num_out, 
        use_bias=False, activation=None, name='regression_head_1'
    )(x)
    """
    NUM_REGRESS_OUT = num_out
    model = Regressor([inp], [x], name='MakiResNet')
    sess = tf.Session()
    model.set_session(sess)

    trainer = AbsTrainer(model, [inp])
    trainer.compile()
    #trainer.set_common_l2_weight_decay(5e-5)

    global_step = tf.Variable(0, dtype=tf.int32)
    #lr = tf.train.exponential_decay(lr, global_step, decay_steps=(size_dataset // batch_size) * 2, decay_rate=0.92)
    lr = tf.train.piecewise_constant_decay(
        global_step, 
        boundaries=[(size_dataset // batch_size) * 7, (size_dataset // batch_size) * 7 + (size_dataset // batch_size) * 7],
        values=[lr, lr * 0.1, lr * 0.01]
    )
    opt = tf.train.AdamOptimizer(lr)
    sess.run(tf.variables_initializer([global_step]))

    return model, trainer, opt, global_step, sess


def get_regression_model(size_hw=(336, 336), batch_size=64, size_dataset=35000, num_out=1, lr=5e-3):
    """
    Create model

    Parameters
    ----------

    Returns
    -------
    model, trainer, opt, global_step, sess

    """
    global NUM_REGRESS_OUT
    inp, x = ResNet18([batch_size, size_hw[0], size_hw[1], 3])
    
    x = GlobalAvgPoolLayer('flat')(x)
    x = ActivationLayer(name='regress_act_0')(x)
    x = BatchNormLayer(D=x.get_shape()[-1], name='regress_bn_0')(x)
    x = DenseLayer(
        in_d=x.get_shape()[-1], out_d=x.get_shape()[-1], 
        use_bias=False, activation=None, name='regression_head_0'
    )(x)
    x = ActivationLayer(name='regress_act_1')(x)
    x = BatchNormLayer(D=x.get_shape()[-1], name='regress_bn_1')(x)
    x = DenseLayer(
        in_d=x.get_shape()[-1], out_d=num_out, 
        use_bias=False, activation=None, name='regression_head_1'
    )(x)
    NUM_REGRESS_OUT = num_out
    model = Regressor([inp], [x], name='MakiResNet')
    sess = tf.Session()
    model.set_session(sess)

    trainer = AbsTrainer(model, [inp])
    trainer.compile()
    #trainer.set_common_l2_weight_decay(5e-5)

    global_step = tf.Variable(0, dtype=tf.int32)
    #lr = tf.train.exponential_decay(lr, global_step, decay_steps=(size_dataset // batch_size) * 2, decay_rate=0.92)
    lr = tf.train.piecewise_constant_decay(
        global_step, 
        boundaries=[(size_dataset // batch_size) * 7, (size_dataset // batch_size) * 7 + (size_dataset // batch_size) * 7],
        values=[lr, lr * 0.1, lr * 0.01]
    )
    opt = tf.train.AdamOptimizer(lr)
    sess.run(tf.variables_initializer([global_step]))

    return model, trainer, opt, global_step, sess


def get_model(size_hw=(336, 336), batch_size=64, size_dataset=10000, num_classes=21, lr=5e-3):
    """
    Create model

    Parameters
    ----------

    Returns
    -------
    model, trainer, opt, global_step, sess

    """
    global NUM_CLASSES
    
    inp, x = ResNet18([batch_size, size_hw[0], size_hw[1], 3])

    x = GlobalAvgPoolLayer('flat')(x)
    # Head1
    x = DropoutLayer(name='drop_final', p_keep=0.75)(x)
    x = DenseLayer(in_d=x.get_shape()[-1], out_d=num_classes, use_bias=False, activation=None, name='classification_head1')(x)
    NUM_CLASSES = num_classes
    model = Classificator(inp, x, name='MakiResNet')
    sess = tf.Session()
    model.set_session(sess)

    trainer = CETrainer(model, [inp])
    trainer.compile()
    trainer.set_common_l2_weight_decay(5e-5)

    global_step = tf.Variable(0, dtype=tf.int32)
    lr = tf.train.exponential_decay(lr, global_step, decay_steps=(size_dataset // batch_size) * 2, decay_rate=0.92)
    opt = tf.train.AdamOptimizer(lr)
    sess.run(tf.variables_initializer([global_step]))

    return model, trainer, opt, global_step, sess


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(A):
    """
    Computes a softmax function. 
    Input: A (N, k) ndarray.
    Returns: (N, k) ndarray.
    """
    e = np.exp(A)
    return e / np.sum(e, axis=-1, keepdims=True)


def predict_regression(input_mf, output_mf, data_pr, batch_size, sess):
    return predict(input_mf, output_mf, data_pr, batch_size, sess, NUM_REGRESS_OUT)

def predict_classification(input_mf, output_mf, data_pr, batch_size, sess):
    return predict(input_mf, output_mf, data_pr, batch_size, sess, NUM_CLASSES)


def predict(input_mf, output_mf, data_pr, batch_size, sess, num_classes):
    ans = [
        output_mf.eval(feed_dict={input_mf: data_pr[i*batch_size:(i+1)*batch_size]}, sess=sess) 
        for i in range(len(data_pr)//batch_size)
    ]
    ans = np.concatenate(ans, axis=0).reshape(-1, num_classes)
    
    return ans.astype(np.float32, copy=False)


def eval_model(model, x_data, y_data, batch_size, calc_mean=True, is_classic=True, return_predict_only=False):
    accur_list = []
    predictions_list = []
    output_mf = model.get_outputs()[0]
    input_mf = model.get_inputs()[0]
    
    for i in range(len(x_data)//batch_size):
        batched_data_x = preprocess_img(x_data[i*batch_size:(i+1)*batch_size])
        batched_data_y = y_data[i*batch_size:(i+1)*batch_size]
        if is_classic:
            predicted_np = np.argmax(
                softmax(
                    predict_classification(
                        input_mf=input_mf,
                        output_mf=output_mf, 
                        data_pr=batched_data_x, 
                        batch_size=batch_size, 
                        sess=model.get_session())
                 ), 
                 axis=-1
            )
            accur_list.append(predicted_np == batched_data_y)
        else:
            predicted_np = predict_regression(
                input_mf=input_mf,
                output_mf=output_mf, 
                data_pr=batched_data_x, 
                batch_size=batch_size, 
                sess=model.get_session()
            )
        predictions_list.append(predicted_np)

    pred_np = np.asarray(predictions_list).reshape(-1)

    if return_predict_only:
        return pred_np

    accur_np = np.asarray(accur_list).reshape(-1)
    if not calc_mean:
        return accur_np, pred_np
    
    return np.mean(accur_list), pred_np


def preprocess_img(data):
    data = np.asarray(data).astype(np.float32, copy=False)
    return coffe_norm(data).astype(np.float32, copy=False)
    #return img_RGB2LAB_preprocess_np(data).astype(np.float32, copy=False)



def generator_wrapper(gen_data, *args):
    while True:
        X_data, Y_data = next(gen_data)
        X_data = X_data[0]
        Y_data = Y_data[0]
        """
        # Aug rotation and flip of the image
        # In my case - images already rotated
        
        rotate_angle = np.random.uniform(-90, 90, size=(len(X_data))).astype(np.float32)
        #X_data = [ndimage.rotate(X_data[i], rotate_angle[i], reshape=False) for i in range(len(X_data))]
        X_data = [
            np.array(
                Image.fromarray(X_data[i]).rotate(rotate_angle[i], resample=Image.BICUBIC).getdata()
            ).reshape(size_hw[0], size_hw[1], 3)
            for i in range(len(X_data))
        ]
        
        ra = random.random()
        type_rotate = None
        
        if ra > 0.3 and ra < 0.5:
            type_rotate = cv2.ROTATE_180
        elif ra > 0.5 and ra < 0.7:
            type_rotate = cv2.ROTATE_90_CLOCKWISE
        elif ra > 0.7:
            type_rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
        
        if type_rotate is not None:
            X_data = [cv2.rotate(X_data[i], type_rotate) for i in range(len(X_data))]
        """
        
        # sigma more than 1.0 - gives very bad looking image, i.e. sigma must be lower than 1.0
        if random.random() > 0.5:
            X_data = [gaussian_filter(X_data[i], sigma=0.5) for i in range(len(X_data))]
        
        if random.random() > 0.5:
            X_data = np.asarray(X_data).astype(np.float32, copy=False)
            X_data += np.random.normal(scale=0.1, size=X_data.shape).astype(np.float32, copy=False)
        
        X_data = preprocess_img(X_data)
        
        yield (X_data,), (Y_data, )


def get_generator_wrapper(Xtrain, Ytrain, batch_size):
    return generator_wrapper(cycle_generator(Xtrain, Ytrain, batch_size))

