import numpy as np
from scipy import linalg

class HARP:
    def __init__(self, a=0.1, c=0.5, c_tilde=0.5, w=0.1, A=100, alpha=0.602, gamma=0.101,
                 iter_num=5000, rep_num=10, direct_num=3,
                 theta_0=None, loss_true=None, loss_noisy=None,
                 record_theta_flag=False, record_loss_flag=True,
                 constraint_flag=True, target_attack_flag=False,
                 n=50, eval_n=1):

        ## step size: a_k = a / (k+1+A) ** alpha
        ## perturbation size: c_k = c / (k+1) ** gamma
        ## direct_num: number of directions per iteration
        self.a = a
        self.c = c
        self.c_tilde = c_tilde
        self.w = w
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

        self.iter_num = iter_num
        self.rep_num = rep_num
        self.direct_num = direct_num

        self.theta_0 = theta_0
        self.p = theta_0.shape[0] # shape = (p,)

        ## initialize perturbation vector
        self.delta_all = np.round(np.random.rand(self.p, direct_num, iter_num, rep_num)) * 2 - 1
        self.delta_tilde_all = np.round(np.random.rand(self.p, direct_num, iter_num, rep_num)) * 2 - 1
        # self.delta_all = np.random.normal(0, 1, (self.p, direct_num, iter_num, rep_num))
        # self.delta_tilde_all = np.random.normal(0, 1, (self.p, direct_num, iter_num, rep_num))

        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        self.constraint_flag = constraint_flag
        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag 
        self.target_attack_flag = target_attack_flag

        self.n = n
        self.eval_n = eval_n

    def get_grad_est(self, theta, iter_idx=0, rep_idx=0):
        c_k = self.c / ((iter_idx + 1) ** self.gamma)
        c_tilde_k = self.c_tilde / (iter_idx + 1) ** self.gamma
        w_k = self.w / (iter_idx + 2)
 
        grad_all = np.empty((self.p, self.direct_num))
        Hess_all = np.empty((self.p, self.p, self.direct_num))
        for direct_idx in range(self.direct_num):
            delta = self.delta_all[:, direct_idx, iter_idx, rep_idx]
            delta_tilde = self.delta_tilde_all[:,direct_idx, iter_idx, rep_idx]

            direct = linalg.solve(self.B, delta).T
            direct_tilde = linalg.solve(self.B, delta_tilde).T 

            ## noisy function measurements
            eval_img_idx = np.random.choice(self.n, self.eval_n, replace = False)
            loss_plus = self.loss_noisy(eval_img_idx, theta + c_k * direct)
            eval_img_idx = np.random.choice(self.n, self.eval_n, replace = False)
            loss_minus = self.loss_noisy(eval_img_idx, theta - c_k * direct)
            eval_img_idx = np.random.choice(self.n, self.eval_n, replace = False)
            loss_plus_tilde = self.loss_noisy(eval_img_idx, theta + c_k * direct + c_tilde_k * direct_tilde)
            eval_img_idx = np.random.choice(self.n, self.eval_n, replace = False)
            loss_minus_tilde = self.loss_noisy(eval_img_idx, theta - c_k * direct + c_tilde_k * direct_tilde)

            ## gradient estimate
            t_direct = np.dot(self.B, delta).T
            loss_diff = loss_plus - loss_minus 
            grad_all[:,direct_idx] = ((loss_plus - loss_minus) / (2 * c_k)) * t_direct

            ## Hessian estimate
            loss_diff = (loss_plus_tilde - loss_minus_tilde) - loss_diff
            t_direct_tilde = np.dot(self.B, delta_tilde).T
            Hess_all[:,:,direct_idx] = (loss_diff / (2 * c_k * c_tilde_k)) * np.dot(t_direct.reshape(self.p,1), t_direct_tilde.reshape(1, self.p))

        grad = np.average(grad_all, axis=1)
        Hess = np.average(Hess_all, axis=2)
        Hess = (Hess + Hess.T) / 2

        self.H_bar = (1 - w_k) * self.H_bar + w_k * Hess
        H_barbar = self.get_pd_matrix_typeA(self.H_bar)
        self.B = self.get_matrix_sqrt(H_barbar)

        return grad

    def get_pd_matrix_typeA(self, H):
        H_eig, H_vec = linalg.eigh(H)
        H_eig_pos = np.maximum(1e-6, np.absolute(H_eig))
        result = np.dot(np.dot(H_vec, np.diag(H_eig_pos)), H_vec.T)
        return (result + result.T) / 2

    # def get_pd_matrix_typeB(self, H):
    #     H_eig, H_vec = linalg.eigh(H)
    #     idx_neg = np.where(H_eig <= 0)
    #     H_eig[idx_neg] = np.maximum(1e-6, np.absolute(H_eig[idx_neg]))
    #     result = np.dot(np.dot(H_vec, np.diag(H_eig)), H_vec.T)
    #     return (result + result.T) / 2

    def get_matrix_sqrt(self, H):
        H_eig, H_vec = linalg.eigh(H)
        return np.dot(np.dot(H_vec, np.diag(np.sqrt(H_eig))), H_vec.T)


    def get_new_est(self, theta, grad, iter_idx=0):
        a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
        return theta - a_k * grad

    # def get_new_est_2nd_order(self, theta, grad, H_barbar, iter_idx=0):
    #     a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
    #     return theta - a_k * linalg.solve(H_barbar, grad)

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

            ## reset estimate at beginning of each replicate
            theta = self.theta_0
            self.H_bar = np.eye(self.p)
            self.B = np.eye(self.p)

            for iter_idx in range(self.iter_num):
                grad = self.get_grad_est(theta, iter_idx, rep_idx)
                theta = self.get_new_est(theta, grad, iter_idx)

                if self.constraint_flag:
                    theta = self.project_est(theta, img_vec)

                ## record result
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
                        print("Algo = %s" % ("HARP"), end=" | ")
                        print("Loss: attack = %3.3f, dist = %3.3f, total = %3.3f" % (attack_loss, dist_loss, total_loss), end=" | ")
                        print("Label: target = %s, adv = %s" % (img_target_label, adv_img_label))
                        print()

                    self.attack_loss_k_all[iter_idx,rep_idx] = attack_loss
                    self.dist_loss_k_all[iter_idx,rep_idx] = dist_loss
                    self.total_loss_k_all[iter_idx,rep_idx] = total_loss
