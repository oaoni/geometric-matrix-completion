#Functional geometric matrix completion
import matplotlib
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.gridspec
from scipy.cluster.hierarchy import linkage

random.seed(0)
np.random.seed(0)

import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy import linalg as linalg
import matplotlib.pyplot as plt
from numpy import linalg as npla
from numpy import matlib
import IPython
plt.rcParams["axes.grid"] = False

from GeometricMatrixCompletion.core import GMC

class FMC(GMC):
    "Functional geometric matrix completion"
    def __init__(self,M,M_training,M_test,S_training,S_test,
                 W_rows,W_cols,p_max,q_max,p_init,q_init,lr,m,n,mat_init=False,
                 reg_param = 0.00001, use_side=True,adam=False,
                 init_vec=[['eye',1],['fmc_default',0],['eye',1]],name='Model'):
        """Constructor"""
        GMC.__init__(self,name)

        self.lr = lr
        self.reg_param = reg_param
        self.m = m
        self.n = n
        self.M = M
        self.S_train = S_training
        self.S_test = S_test
        self.adam = adam
        self.mat_init = mat_init
        self.mu = 'n/a'
        self.rho = 'n/a'
        self.alpha = 'n/a'


        if use_side:
            L_rows, L_cols = self._compute_laplacians(W_rows, W_cols)
            eig_vals_row, eig_vecs_row, eig_vals_col, eig_vecs_col = self._compute_eigs(L_rows, L_cols)
        else:
            eig_vecs_row = np.zeros((M.shape[0], m))
            eig_vecs_col = np.zeros((M.shape[1], n))
            eig_vals_row = np.zeros(m)
            eig_vals_col = np.zeros(n)


        g = self.tf_graph
        with g.as_default():
            #Build model
            self.build_model(M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                        eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n,init_vec)


            # Create a session for running Ops on the Graph for the given model.
            self.config = tf.ConfigProto(allow_soft_placement=True)
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.config)
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())


    def build_model(self,M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                    eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n,init_vec):
        """Create graph"""

        lr = tf.placeholder(tf.float32, name="lr")

        p_init,c_init,q_init = init_vec

        P_tf = self._initialize_dmf_var(p_init,[m,p_max],eig_vecs_row,eig_vecs_col,M_training,'P')
        C_tf = self._initialize_dmf_var(c_init,[p_max,q_max],eig_vecs_row,eig_vecs_col,M_training,'C')
        Q_tf = self._initialize_dmf_var(q_init,[n,q_max],eig_vecs_row,eig_vecs_col,M_training,'Q')

        C_new = tf.matmul(tf.matmul(P_tf, C_tf), tf.transpose(Q_tf)) #check

        Phi_tf = tf.constant(eig_vecs_row[:,0:m], dtype=tf.float32)
        Psi_tf = tf.constant(eig_vecs_col[:,0:n], dtype=tf.float32)

        lambda_row_tf = tf.constant(eig_vals_row[0:m], dtype=tf.float32)
        lambda_col_tf = tf.constant(eig_vals_col[0:n], dtype=tf.float32)
        S_training_tf = tf.constant(S_training, dtype=tf.float32)
        S_test_tf = tf.constant(S_test, dtype=tf.float32)
        M_training_tf = tf.constant(M_training, dtype=tf.float32)
        M_test_tf = tf.constant(M_test, dtype=tf.float32)

        self.X = tf.matmul(tf.matmul(Phi_tf, C_new), tf.transpose(Psi_tf))#reconstruction

        E_data = self._squared_frobenius_norm(tf.multiply(self.X, S_training) - M_training)

        left_mul = tf.matmul(C_new, tf.diag(lambda_col_tf))
        right_mul = tf.matmul(tf.diag(lambda_row_tf),C_new)
        E_comm = self._squared_frobenius_norm(left_mul-right_mul)

        E_tot = E_data + self.reg_param*E_comm
        if self.adam:
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.opt_op = optimizer.minimize(E_tot)

        # Metrics
        self.train_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_training_tf) - M_training_tf)/ tf.reduce_sum(S_training_tf))
        # validation_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(S_validation_tf, (X - M))) / tf.reduce_sum(S_validation_tf))
        self.test_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_test_tf)- M_test_tf)/tf.reduce_sum(S_test_tf))

        ground = tf.placeholder(tf.float32, name='ground')
        mask = tf.placeholder(tf.float32, name='mask')
        self.corr_metric = self._corr_metric(ground=ground, pred=self.X, mask=mask)

class FMCv2(GMC):
    "Functional geometric matrix completion"
    def __init__(self,M,M_training,M_test,S_training,S_test,
                 W_rows,W_cols,p_max,q_max,p_init,q_init,lr,m,n,mat_init=False,
                 init_rank=False,reg_param = 0.00001, use_side=True,adam=False,name='Model'):
        """Constructor"""
        GMC.__init__(self,name)

        self.lr = lr
        self.reg_param = reg_param
        self.m = m
        self.n = n
        self.M = M
        self.S_train = S_training
        self.S_test = S_test
        self.adam = adam
        self.mu = 'n/a'
        self.rho = 'n/a'
        self.alpha = 'n/a'


        if use_side:
            L_rows, L_cols = self._compute_laplacians(W_rows, W_cols)
            eig_vals_row, eig_vecs_row, eig_vals_col, eig_vecs_col = self._compute_eigs(L_rows, L_cols)
        else:
            eig_vecs_row = np.zeros((M.shape[0], m))
            eig_vecs_col = np.zeros((M.shape[1], n))
            eig_vals_row = np.zeros(m)
            eig_vals_col = np.zeros(n)


        g = self.tf_graph
        with g.as_default():
            #Build model
            self.build_model(M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                        eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n,mat_init,init_rank)


            # Create a session for running Ops on the Graph for the given model.
            self.config = tf.ConfigProto(allow_soft_placement=True)
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.config)
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())


    def build_model(self,M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                    eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n,mat_init,init_rank):
        """Create v2 graph"""

        lr = tf.placeholder(tf.float32, name="lr")

        if isinstance(mat_init,np.ndarray):

            U,S,Vh = np.linalg.svd(mat_init, full_matrices =False)

            P_init = U[:,0:init_rank]

            C_init = np.diag(S[0:init_rank])

            Q_init = Vh[0:init_rank,:].T
        else:

            P_init = np.eye(p_max, m)

            C_init = np.zeros([m, m], dtype = np.float32)
            C_init[p_init-1,q_init-1] = np.matmul(np.matmul(np.transpose(eig_vecs_row[:, 0:p_init]),M_training), eig_vecs_col[:, 0:q_init])

            Q_init = np.eye(q_max, m)


        C_tf = tf.Variable(C_init, trainable=True, dtype=tf.float32)
        P_tf = tf.Variable(P_init, trainable=True, dtype=tf.float32)
        Q_tf = tf.Variable(Q_init, trainable=True, dtype=tf.float32)
        C_new = tf.matmul(tf.matmul(P_tf, C_tf), tf.transpose(Q_tf)) #check

        Phi_tf = tf.constant(eig_vecs_row, dtype=tf.float32)
        Psi_tf = tf.constant(eig_vecs_col, dtype=tf.float32)
        lambda_row_tf = tf.constant(eig_vals_row, dtype=tf.float32)
        lambda_col_tf = tf.constant(eig_vals_col, dtype=tf.float32)

        S_training_tf = tf.constant(S_training, dtype=tf.float32)
        S_test_tf = tf.constant(S_test, dtype=tf.float32)
        M_training_tf = tf.constant(M_training, dtype=tf.float32)
        M_test_tf = tf.constant(M_test, dtype=tf.float32)
        self.X = tf.matmul(tf.matmul(Phi_tf, C_new), tf.transpose(Psi_tf))#reconstruction

        E_data = self._squared_frobenius_norm(tf.multiply(self.X, S_training) - M_training)

        C_new_t = tf.transpose(C_new)
        left_mul = tf.matmul(C_new, tf.diag(lambda_col_tf))
        right_mul = tf.matmul(tf.diag(lambda_row_tf),C_new)
        E_comm = self._squared_frobenius_norm(left_mul-right_mul)

        E_tot = E_data + self.reg_param*E_comm
        if self.adam:
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.opt_op = optimizer.minimize(E_tot)

        # Metrics
        self.train_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_training_tf) - M_training_tf)/ tf.reduce_sum(S_training_tf))
        # validation_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(S_validation_tf, (X - M))) / tf.reduce_sum(S_validation_tf))
        self.test_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_test_tf)- M_test_tf)/tf.reduce_sum(S_test_tf))

        ground = tf.placeholder(tf.float32, name='ground')
        mask = tf.placeholder(tf.float32, name='mask')

        self.corr_metric = self._corr_metric(ground=ground, pred=self.X, mask=mask)
