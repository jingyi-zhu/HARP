import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
import tensorflow as tf
import random

from keras.layers import Lambda
from tqdm import tqdm

import loss.image_attack as image_attack
import algorithm.harp as harp
import algorithm.zo_adamm as zo_adamm
import algorithm.zo_scd as zo_scd

def loss_true(adv_delta):
  attack_loss, dist_loss, total_loss = img_attack.get_loss_value(adv_delta)
  img_target_label = img_attack.get_img_target_label()
  adv_img_label = img_attack.get_adv_img_label()
  return attack_loss, dist_loss, total_loss, img_target_label, adv_img_label

def loss_noisy(eval_img_idx, adv_delta):
  _, _, total_loss = img_attack.get_noisy_loss_value(eval_img_idx, adv_delta)
  return total_loss


orig_label = 1 # within which cluster to perform universal attack
rep_num = 25; # replicate number
iter_num = 1000; # iter number within one replicate
query_num = 60; # zeroth-order query within one iter
n = 100 # number of images within specified cluster for universal attack
eval_n = 1 # number of images evaluated in loss_noisy
constraint_flag = True
target_attack_flag = False

## Algorithms and Hyper-parameters
algo = "HARP"; a = 0.005; c = 0.04; w = 1e-5;
# algo = "ZoAdaMM"; a = 0.5; c = 0.04; 
# algo = "ZoSCD"

## Datasets
dataset = "mnist" 
# dataset = "cifar10"

SEED = 1
with tf.Session() as sess:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    ## initialization
    img_attack = image_attack.ImageAttack(dataset, sess, constraint_flag=constraint_flag, target_attack_flag=target_attack_flag)
    adv_delta_0 = np.zeros(img_attack.p) # initial adversarial perturbation is all zeros

    ## algorithm
    if algo == "HARP":
        solver = harp.HARP(a=a, c=c, c_tilde=c, w=w,
                           iter_num=iter_num, rep_num=rep_num, direct_num=int(query_num/4),
                           theta_0=adv_delta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                           record_theta_flag=False, record_loss_flag=True,
                           constraint_flag=constraint_flag, target_attack_flag=target_attack_flag,
                           n=n, eval_n=eval_n)
    elif algo == "ZoAdaMM":
        solver = zo_adamm.ZoAdaMM(a=a, c=c,
                                  iter_num=iter_num, rep_num=rep_num, direct_num=int(query_num/2),
                                  theta_0=adv_delta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                                  record_theta_flag=False, record_loss_flag=True,
                                  constraint_flag=constraint_flag, target_attack_flag=target_attack_flag,
                                  n=n, eval_n=eval_n)
    elif algo == "ZoSCD":
        solver = zo_scd.ZoSCD(a=a, c=c,
                              iter_num=iter_num, rep_num=rep_num, direct_num=int(query_num/2),
                              theta_0=adv_delta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                              record_theta_flag=False, record_loss_flag=True,
                              constraint_flag=constraint_flag, target_attack_flag=target_attack_flag,
                              n=n, eval_n=eval_n)

    img_idx_by_label = img_attack.get_image_by_label(orig_label)
    img_id_all = np.random.choice(img_idx_by_label, n*3, replace=False) 

    img_idx = 0 
    img_id_all_good = [] 
    for img_id in img_id_all:
        img_attack.get_image_by_id([img_id])  
        img_label = img_attack.get_img_label()
        img_pred_label = img_attack.get_img_pred_label()
        if (img_label == img_pred_label):  
           img_id_all_good.append(img_id)
           img_idx += 1
           if img_idx == n:
               break
    print(img_id_all_good)

    img_attack.get_image_by_id(img_id_all_good)  
    img_label = img_attack.get_img_label()
    if img_attack.target_attack_flag:
        img_target_label = (np.array(img_label) + 1) % 10
    else:
        img_target_label = img_label
    img_attack.set_image_target_label(img_target_label)
            
    print("="*100)
    print("image number:", n, "img_id:", img_id_all_good, "img_label:", img_label, "target_label", img_target_label)

    img_vec = img_attack.get_img_vec() # projection 

    solver.train(img_vec)
    attack_loss_all = solver.attack_loss_k_all
    dist_loss_all = solver.dist_loss_k_all
    total_loss_all = solver.total_loss_k_all


    suffix0 = "universal/cluster_{}_algo_{}_rep_{}_iter_{}_query_{}_a_{}_c_{}_w_{}_N_{}_n_{}".format(orig_label, algo, str(rep_num), str(iter_num), str(query_num), str(a), str(c), str(w), str(n), str(eval_n))
    np.savez("{}".format(suffix0), total_loss_all=total_loss_all, distortion_loss_all = dist_loss_all, attack_loss_all = attack_loss_all)         