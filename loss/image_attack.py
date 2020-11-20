import os
import sys
import tensorflow as tf
import numpy as np
import random

from keras.layers import Lambda
from tqdm import tqdm

from loss.setup_mnist import MNIST, MNISTModel
from loss.setup_cifar import CIFAR, CIFARModel
import loss.Utils as util

class ImageAttack:
    def __init__(self, dataset=None, sess=None,
                 constraint_flag=True, target_attack_flag=False):
        # sess: tensorflow session
        if dataset == "mnist":
            self.data = MNIST()
            self.model = MNISTModel("loss/models/mnist", sess, True)
        elif dataset == "cifar10":
            self.data = CIFAR()
            self.model = CIFARModel("loss/models/cifar", sess, True)
        else:
            print('Please specify a valid dataset.')

        self.regularizer = 10
        self.kappa = 1e-10

        self.p = self.data.test_data[0].size # dimension
        self.constraint_flag = constraint_flag
        self.target_attack_flag = target_attack_flag

    def get_image_by_label(self, image_label):
        img_idx_by_label = np.where(np.argmax(self.data.test_labels, 1) == image_label)
        return img_idx_by_label[0]
        
    def get_image_by_id(self, image_id):

        # image_id = [5465]

        # image_ids: list of image id
        self.n = len(image_id)
        self.img = self.data.test_data[image_id] # shape = (n,28,28,1)
        self.img_label = np.argmax(self.data.test_labels[image_id], 1) # list[n]

        img_prob, _, _ = util.model_prediction(self.model, self.img)
        self.img_pred_label = np.argmax(img_prob, 1)

        # print(img_prob)
        # raise

    def set_image_target_label(self, target_label):
        # target_labels: list of target lable
        self.img_target_label = target_label # list[n]

    def get_img_vec(self):
        return np.resize(self.img, (self.n, self.p))

    def get_adv_img_vec(self, img_vec, adv_delta):
        if self.constraint_flag:
            adv_img_vec = np.clip(img_vec + adv_delta, -0.5, 0.5)
        else:
            adv_img_vec = np.arctanh(2 * img_vec * 0.999999) + adv_delta
            adv_img_vec = 0.5 * np.tanh(adv_img_vec) / 0.999999
        return adv_img_vec

    def get_loss_value(self, adv_delta):
        img_vec = self.get_img_vec()
        adv_img_vec = self.get_adv_img_vec(img_vec, adv_delta)
        adv_img = np.resize(adv_img_vec, self.img.shape)
        adv_img_prob, _, _ = util.model_prediction(self.model, adv_img)

        self.adv_img_label = np.argmax(adv_img_prob, 1) # for printing result

        temp = adv_img_prob.copy()
        temp[np.arange(self.n), self.img_target_label] = 0

        if self.target_attack_flag:
            attack_loss = np.maximum(np.log(np.amax(temp,1) + 1e-10) - np.log(adv_img_prob[np.arange(self.n), self.img_target_label] + 1e-10),
                            - np.ones(self.n) * self.kappa)
        else:
            attack_loss = np.maximum(np.log(adv_img_prob[np.arange(self.n), self.img_target_label] + 1e-10) - np.log(np.amax(temp,1) + 1e-10),
                            - np.ones(self.n) * self.kappa)

        attack_loss = self.regularizer * np.sum(attack_loss) / self.n
        dist_loss = np.linalg.norm(adv_img_vec[0] - img_vec[0]) ** 2

        total_loss = attack_loss + dist_loss
        return attack_loss, dist_loss, total_loss

    def get_noisy_loss_value(self, eval_img_idx, adv_delta):
        eval_img_idx_sort = np.sort(eval_img_idx)
        # print(eval_img_idx_sort)
        eval_img_n = len(eval_img_idx_sort) # eval_img_idx_sort = sorted random samples from 0, ..., n-1
        
        img_vec = self.get_img_vec() # shape = (n,784)
        eval_img_vec = img_vec[eval_img_idx_sort,:] # shape = (eval_img_n,784)
        adv_eval_img_vec = self.get_adv_img_vec(eval_img_vec, adv_delta) # shape = (eval_img_n, 784)
        
        eval_img_shape = (eval_img_n,) + self.img.shape[1:] # shape = (eval_img_n, 28, 28, 1)
        adv_eval_img = np.resize(adv_eval_img_vec, eval_img_shape)
        adv_eval_img_prob, _, _ = util.model_prediction(self.model, adv_eval_img)
        # adv_eval_img_label = np.argmax(adv_eval_img_prob, 1) # for printing result

        eval_temp = adv_eval_img_prob.copy() # shape = (eval_img_n, 10)
        eval_img_target_label = self.img_target_label[eval_img_idx_sort]
        eval_temp[np.arange(eval_img_n), eval_img_target_label] = 0

        if self.target_attack_flag:
            attack_loss = np.maximum(np.log(np.amax(eval_temp,1) + 1e-10) - np.log(adv_eval_img_prob[np.arange(eval_img_n), eval_img_target_label] + 1e-10),
                            - np.ones(eval_img_n) * self.kappa)
        else:
            attack_loss = np.maximum(np.log(adv_eval_img_prob[np.arange(eval_img_n), eval_img_target_label] + 1e-10) - np.log(np.amax(eval_temp,1) + 1e-10),
                            - np.ones(eval_img_n) * self.kappa)

        attack_loss = self.regularizer * np.sum(attack_loss) / eval_img_n
        dist_loss = np.linalg.norm(adv_eval_img_vec[0] - eval_img_vec[0]) ** 2
        total_loss = attack_loss + dist_loss

        # # TESTING
        # print("noisy loss value:", attack_loss)
        # attack_loss_true, _, _ = self.get_loss_value(adv_delta)
        # print("noisy loss true:", attack_loss_true)

        return attack_loss, dist_loss, total_loss

    # for printing result
    def get_img_label(self):
        return self.img_label
    
    def get_img_pred_label(self):
        return self.img_pred_label
    
    def get_img_target_label(self):
        return self.img_target_label
    
    def get_adv_img_label(self):
        return self.adv_img_label
