import numpy as np
import matplotlib
import os
import tensorflow as tf
#import numpy as np
import random
random.seed(0)
np.random.seed(0) #fix seed so we can reproduce the results
import h5py
import scipy
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy import linalg as linalg
import matplotlib.pyplot as plt
import networkx as nx
from numpy import linalg as npla
import IPython
plt.rcParams["axes.grid"] = False

from GeometricMatrixCompletion.core import GMC

class SGMC(GMC):
    "Structural geometric matrix completion"
    def __init__(self,M,M_training,M_test,S_training,S_test,
                 W_rows,W_cols,p_max,q_max,p_init,q_init,lr,m,n,
                 mu_r,mu_c,rho_r,rho_c,alpha,mat_init=False,
                 adam=False,DMF=False,name='Model'):
        """Constructor"""
        GMC.__init__(self,name)

        self.lr = lr
        self.m = m
        self.n = n
        self.M = M
        self.S_train = S_training
        self.S_test = S_test
        self.adam = adam
        self.mat_init = mat_init



        L_rows, L_cols = self._compute_laplacians(W_rows, W_cols)
        eig_vals_row, eig_vecs_row, eig_vals_col, eig_vecs_col = self._compute_eigs(L_rows, L_cols)


        g = self.tf_graph
        with g.as_default():
            #Build model
            self.build_model(M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                        eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n,
                        mu_r,mu_c,rho_r,rho_c,alpha,DMF)


            # Create a session for running Ops on the Graph for the given model.
            self.config = tf.ConfigProto(allow_soft_placement=True)
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.config)
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())

    def build_model(self,M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                    eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n,
                    mu_r,mu_c,rho_r,rho_c,alpha,DMF):
        """Create sgmc v0 graph"""

        #define flags
        perturb_flag = False
        zoomout_flag = False

        # set flags for trainable parameters
        train_P_flag = True
        train_Q_flag = True
        train_C_flag = True
        train_lambda_row_flag = False
        train_lambda_col_flag = False

        if perturb_flag:
            # perturb graphs and use the perturbed graphs for Phi, Psi

            noise_level = 0.15

            # Perturb graphs
            row_noise = np.random.uniform(low=-noise_level, high=noise_level, size=[m, m])
            row_noise = 0.5 * (row_noise + np.transpose(row_noise))
            W_rows_perturbed = np.maximum(W_rows + row_noise, 0)

            col_noise = np.random.uniform(low=-noise_level, high=noise_level, size=[n, n])
            col_noise = 0.5 * (col_noise + np.transpose(col_noise))
            W_cols_perturbed = np.maximum(W_cols + col_noise, 0)


            #visualize perturbed graphs

            G_rows_perturbed = nx.from_numpy_array(W_rows_perturbed)
            G_cols_perturbed = nx.from_numpy_matrix(W_cols_perturbed)

            # plot graphs
            plt.subplot(1, 3, 1)
            nx.draw(G_rows_perturbed)
            plt.title('G_rows_perturbed')

            plt.subplot(1, 3, 2)
            nx.draw(G_cols_perturbed)
            plt.title('G_cols_perturbed')

            eig_vals_row, eig_vecs_row = self._init_graph_basis(W_rows_perturbed)
            eig_vals_col, eig_vecs_col = self._init_graph_basis(W_cols_perturbed)

            eig_vals_row = np.expand_dims(eig_vals_row,1)
            eig_vals_col = np.expand_dims(eig_vals_col,1)

            eig_vals_row_orig, eig_vecs_row_orig = self._init_graph_basis(W_rows)

            plt.subplot(1, 3, 3)
            plt.plot(eig_vals_row, 'r')
            plt.plot(eig_vals_row_orig, 'g')
            plt.show()

        if DMF:
            # for Deep Matrix Factorization, use Identity basis

            eig_vals_row, eig_vecs_row = np.ones(m, dtype=np.float32), np.eye(m,dtype=np.float32)
            eig_vals_col, eig_vecs_col = np.ones(n, dtype=np.float32), np.eye(n,dtype=np.float32)

            eig_vals_row = np.expand_dims(eig_vals_row,1)
            eig_vals_col = np.expand_dims(eig_vals_col,1)
            alpha = 2   # In DMF the initialization should be close to zero, so we use 10^{-2}

            #for DMF all flags should be true
            train_P_flag = True
            train_Q_flag = True
            train_C_flag = True
            train_lambda_row_flag = False
            train_lambda_col_flag = False

        lr = tf.placeholder(tf.float32, name="lr")

        # initialize C,P,Q
        P_init = (10**(-alpha))*np.eye(m, p_max)
        C_init = (10**(-alpha))*np.eye(p_max, q_max)
        Q_init = (10**(-alpha))*np.eye(n, q_max)

        P_tf = tf.Variable(P_init, trainable=train_P_flag, dtype=tf.float32)
        C_tf = tf.Variable(C_init, trainable=train_C_flag, dtype=tf.float32)
        Q_tf = tf.Variable(Q_init, trainable=train_Q_flag, dtype=tf.float32)

        Phi_tf = tf.constant(eig_vecs_row, dtype=tf.float32)
        Psi_tf = tf.constant(eig_vecs_col, dtype=tf.float32)

        lambda_row_tf = tf.Variable(eig_vals_row.reshape(-1,1), trainable=train_lambda_row_flag,
                                    dtype=tf.float32)
        lambda_col_tf = tf.Variable(eig_vals_col.reshape(-1,1), trainable=train_lambda_col_flag,
                                    dtype=tf.float32)

        S_training_tf = tf.constant(S_training, dtype=tf.float32)
        S_test_tf = tf.constant(S_test, dtype=tf.float32)
        M_training_tf = tf.constant(M_training, dtype=tf.float32)
        M_test_tf = tf.constant(M_test, dtype=tf.float32)

        C_new = tf.matmul(tf.matmul(P_tf, C_tf), tf.transpose(Q_tf))
        self.X = tf.matmul(tf.matmul(Phi_tf, C_new), tf.transpose(Psi_tf))
        X = self.X
        E_data = self._squared_frobenius_norm(tf.multiply(self.X, S_training_tf) - M_training)

        # estimate singular values of X
        sing_vals, u_tf, v_tf = tf.svd(self.X, compute_uv=True, full_matrices=True)
        nuclear_norm = tf.norm(sing_vals, ord=1)

        # Dirichlet energy
        C_new_t = tf.transpose(C_new)
        E_dirichlet_rows = tf.reduce_sum(tf.matmul(tf.transpose(tf.nn.relu(lambda_row_tf[0:C_new.shape[0],:])), tf.multiply(C_new, C_new)))
        E_dirichlet_cols = tf.reduce_sum(tf.matmul(tf.transpose(tf.nn.relu(lambda_col_tf[0:C_new_t.shape[0],:])), tf.multiply(C_new_t, C_new_t)))

        # off diag energy
        off_diag_mask_rows = tf.ones([p_max, p_max]) - tf.eye(p_max)
        off_diag_mask_cols = tf.ones([q_max, q_max]) - tf.eye(q_max)

        diagonalized_rows = tf.matmul(tf.matmul(tf.transpose(P_tf), tf.diag(tf.squeeze(tf.nn.relu(lambda_row_tf)))), P_tf)
        diagonalized_cols = tf.matmul(tf.matmul(tf.transpose(Q_tf), tf.diag(tf.squeeze(tf.nn.relu(lambda_col_tf)))), Q_tf)

        E_diag_rows = self._squared_frobenius_norm(tf.multiply(off_diag_mask_rows, diagonalized_rows))
        E_diag_cols = self._squared_frobenius_norm(tf.multiply(off_diag_mask_cols, diagonalized_cols))


        if DMF:
            E_tot = E_data
        else:
            E_tot = E_data + mu_r*E_dirichlet_rows+mu_c*E_dirichlet_cols+rho_r*E_diag_rows+rho_c*E_diag_cols

        # get gradients
        dL_dX = tf.gradients(E_tot, self.X)
        diag_X = tf.eye(M_training.shape[0], M_training.shape[1])
        off_diag_X = tf.ones(M_training.shape) - diag_X
        grad_uv = tf.abs(tf.matmul(tf.transpose(u_tf), tf.matmul(dL_dX[0], v_tf)))
        diag_mean = tf.reduce_mean(tf.multiply(grad_uv, diag_X))
        diag_std = tf.math.reduce_std(tf.multiply(grad_uv, diag_X))
        off_diag_mean = tf.reduce_mean(tf.multiply(grad_uv, off_diag_X))
        off_diag_std = tf.math.reduce_std(tf.multiply(grad_uv, off_diag_X))

        if self.adam:
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.opt_op = optimizer.minimize(E_tot)

        # Create a session for running ops on the Graph.
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        self.train_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_training_tf) - M_training_tf) / tf.reduce_sum(S_training_tf))
        self.test_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_test_tf)- M_test_tf) / tf.reduce_sum(S_test_tf))

        ground = tf.placeholder(tf.float32, name='ground')
        mask = tf.placeholder(tf.float32, name='mask')

        self.corr_metric = self._corr_metric(ground=ground, pred=self.X, mask=mask)
