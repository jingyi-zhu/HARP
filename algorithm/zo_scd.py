import numpy as np
from scipy import linalg

class ZoSCD:
    def __init__(self, a=3e-4, c=5e-3, A=0, alpha=0.602, gamma=0.101,
                 iter_num=100, rep_num=1, direct_num=11,
                 theta_0=None, loss_true=None, loss_noisy=None,
                 record_theta_flag=False, record_loss_flag=True,
                 constraint_flag=True, target_attack_flag=False,
                 n=50, eval_n=1):

        # step size: a_k = a / (k+1+A) ** alpha
        # perturbation size: c_k = c / (k+1) ** gamma
        # direct_num: number of directions per iteration

        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.direct_num = direct_num

        self.theta_0 = theta_0
        self.p = theta_0.shape[0] # shape = (p,)
        
        # initialize perturbation vector
        self.delta_idx_all = np.random.randint(self.p, size=(direct_num, iter_num, rep_num))

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        self.constraint_flag = constraint_flag
        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

        self.target_attack_flag = target_attack_flag

        self.n = n
        self.eval_n = eval_n

    def get_grad_est(self, theta, iter_idx=0, rep_idx=0):
        eval_img_idx = np.random.choice(self.n, self.eval_n, replace = False)

        c_k = self.c / (iter_idx + 1) ** self.gamma
        grad_all = np.empty((self.p, self.direct_num))
        for direct_idx in range(self.direct_num):
            delta_idx = self.delta_idx_all[direct_idx, iter_idx, rep_idx]

            direct = np.zeros(self.p)
            direct[delta_idx] = 1

            loss_plus = self.loss_noisy(eval_img_idx, theta + c_k * direct)
            loss_minus = self.loss_noisy(eval_img_idx, theta - c_k * direct)
            grad_all[:,direct_idx] = (loss_plus - loss_minus) / (2 * c_k) * direct * self.p

        grad = np.average(grad_all, axis=1)
        return grad

    def get_new_est(self, theta, grad, iter_idx=0):
        a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
        return theta - a_k * grad

    def project_est(self, delta_adv, img_vec):
        a_point = delta_adv.reshape((1,self.p))
        X = img_vec
        Vt = np.ones((1,self.p))

        VtX = np.sqrt(Vt) * X
    
        min_VtX = np.min(VtX, axis=0)
        max_VtX = np.max(VtX, axis=0)

        lb = -0.5 * np.sqrt(Vt) - min_VtX
        ub = 0.5 * np.sqrt(Vt) - max_VtX
        
        a_temp = np.sqrt(Vt) * a_point
        z_proj_temp = np.multiply(lb, np.less(a_temp, lb)) + np.multiply(ub, np.greater(a_temp, ub)) \
                      + np.multiply(a_temp, np.multiply(np.greater_equal(a_temp, lb), np.less_equal(a_temp, ub)))
        delta_proj = 1/np.sqrt(Vt) * z_proj_temp # delta_proj = np.diag(1/np.diag(np.sqrt(Vt)))*z_proj_temp
        return delta_proj.reshape(a_point.shape)

    def train(self, img_vec):
        if self.record_theta_flag:
            self.theta_k_all = np.empty((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.attack_loss_k_all = np.empty((self.iter_num, self.rep_num))
            self.dist_loss_k_all = np.empty((self.iter_num, self.rep_num))
            self.total_loss_k_all = np.empty((self.iter_num, self.rep_num))

        for rep_idx in range(self.rep_num):
            print("rep_idx:", rep_idx, "/", self.rep_num)

            # reset estimate
            theta = self.theta_0

            for iter_idx in range(self.iter_num):
                grad = self.get_grad_est(theta, iter_idx, rep_idx)
                theta = self.get_new_est(theta, grad, iter_idx)

                if self.constraint_flag:
                    theta = self.project_est(theta, img_vec)

                # record result
                if self.record_theta_flag:
                    self.theta_k_all[:,iter_idx,rep_idx] = theta
                if self.record_loss_flag:
                    attack_loss, dist_loss, total_loss, img_target_label, adv_img_label = self.loss_true(theta)

                    if (iter_idx % 10) == 0:
                        if self.target_attack_flag:
                            succ_rate = np.average(img_target_label == adv_img_label)
                        else:
                            succ_rate = np.average(img_target_label != adv_img_label)
                        
                        print("Iter = %d, succ rate = %.2f" % (iter_idx, succ_rate), end=" | ")
                        print("Algo = %s" % ("ZoSCD"), end=" | ")
                        print("Loss: attack = %3.3f, dist = %3.3f, total = %3.3f" % (attack_loss, dist_loss, total_loss), end=" | ")
                        print("Label: target = %s, adv = %s" % (img_target_label, adv_img_label))
                        print()

                    self.attack_loss_k_all[iter_idx,rep_idx] = attack_loss
                    self.dist_loss_k_all[iter_idx,rep_idx] = dist_loss
                    self.total_loss_k_all[iter_idx,rep_idx] = total_loss