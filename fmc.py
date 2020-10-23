#Functional geometric matrix completion
import matplotlib
import tensorflow as tf
import numpy as np
import random

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

class FMC:
    "Functional geometric matrix completion"
    def __init__(self,M_training,M_test,S_training,S_test,
                 W_rows,W_cols,p_max,q_max,p_init,q_init,lr,m,n):
        
        self.lr = lr
        self.m = m
        self.n = n
        
        L_rows, L_cols = self._compute_laplacians(W_rows, W_cols)
        eig_vals_row, eig_vecs_row, eig_vals_col, eig_vecs_col = self._compute_eigs(L_rows, L_cols)
        
        #Build model
        self.build_model(M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                    eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n)

        # Create a session for running Ops on the Graph.
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        
    def build_model(self,M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                    eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n):
        """Create graph"""
        
        C_init = np.zeros([p_max, q_max], dtype = np.float32)
        #C_init = np.zeros([m, n], dtype = np.float32)
        C_init[p_init-1,q_init-1] = np.matmul(np.matmul(np.transpose(eig_vecs_row[:, 0:p_init]),M_training), eig_vecs_col[:, 0:q_init])

        P_init = np.eye(m, p_max)
        Q_init = np.eye(n, q_max)

        C_tf = tf.Variable(C_init, trainable=True, dtype=tf.float32)
        #C_tf = tf.Variable(np.matmul(np.matmul(P_init, C_init), np.transpose(Q_init)), trainable=True, dtype=tf.float32)
        P_tf = tf.Variable(P_init, trainable=True, dtype=tf.float32)
        Q_tf = tf.Variable(Q_init, trainable=True, dtype=tf.float32)
        C_new = tf.matmul(tf.matmul(P_tf, C_tf), tf.transpose(Q_tf)) #check
        #C_new = C_tf
        Phi_tf = tf.constant(eig_vecs_row[:,0:m], dtype=tf.float32)
        Psi_tf = tf.constant(eig_vecs_col[:,0:n], dtype=tf.float32)

        lambda_row_tf = tf.constant(eig_vals_row[0:m], dtype=tf.float32)
        lambda_col_tf = tf.constant(eig_vals_col[0:n], dtype=tf.float32)

        S_training_tf = tf.constant(S_training, dtype=tf.float32)
        S_test_tf = tf.constant(S_test, dtype=tf.float32)
        M_training_tf = tf.constant(M_training, dtype=tf.float32)
        M_test_tf = tf.constant(M_test, dtype=tf.float32)
        self.X = tf.matmul(tf.matmul(Phi_tf, C_new), tf.transpose(Psi_tf))

        E_data = self._squared_frobenius_norm(tf.multiply(self.X, S_training) - M_training)

        C_new_t = tf.transpose(C_new)
        left_mul = tf.matmul(C_new, tf.diag(lambda_col_tf))
        right_mul = tf.matmul(tf.diag(lambda_row_tf),C_new)
        E_comm = self._squared_frobenius_norm(left_mul-right_mul)

        E_tot = E_data + .00001*E_comm
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.opt_op = optimizer.minimize(E_tot)
        
        self.train_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_training_tf) - M_training_tf)/ tf.reduce_sum(S_training_tf))
        #validation_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(S_validation_tf, (X - M))) / tf.reduce_sum(S_validation_tf))
        self.test_loss = tf.sqrt(self._ squared_frobenius_norm(tf.multiply(self.X, S_test_tf)- M_test_tf)/tf.reduce_sum(S_test_tf))

    def fit(self, num_iters, grid=[], plot=False):
        
        
        train_loss_log = []
        test_loss_log = []
        iter_log = []
        

        for iter in range(num_iters):
            
            
            if iter%1000 == 0:
                train_loss_np, test_loss_np = self.sess.run([self.train_loss, self.test_loss])
                train_loss_log.append(train_loss_np)
                test_loss_log.append(test_loss_np)
                iter_log.append(iter)
                if (plot) and (iter%5000 == 0):
                    IPython.display.clear_output()    
                    print("iter " + str(iter) +" ,train loss: "+str(train_loss_np)+", test loss: " + str(test_loss_np) )
                    print("Hyperparameters: lr: {}, m: {}, n: {}".format(self.lr, self.m, self.n))
                    X_np = self.sess.run(self.X)   
                    fig, ax = plt.subplots(1,3, figsize=(15,5))
                    ax[0].imshow(M)
                    ax[0].set_title("True")
                    ax[1].imshow(X_np)
                    ax[1].set_title("X")
                    ax[2].plot(iter_log[3:], train_loss_log[3:], 'r', iter_log[3:], test_loss_log[3:], 'b')
                    ax[2].set_title("Train and Test Loss")
                    plt.legend(['Train Loss','Test Loss'])
                    plt.show()
                    display(grid)
                    
                summary_dic = {'learning_rate':self.lr,'m':self.m,'n':self.n,'num_iters':num_iters,
                               'train_loss':train_loss_log[-1],'test_loss':test_loss_log[-1],
                               'best_train_loss':np.array(train_loss_log).min(),'best_test_loss':np.array(test_loss_log).min()}

                self.summary_dic = summary_dic
                    
            self.sess.run(self.opt_op)
            
    def _compute_laplacians(self, W_rows, W_cols):
        # produce Laplacians of the row and column graphs
        L_rows = self._init_graph_basis(W_rows)
        L_cols = self._init_graph_basis(W_cols)
        
        return L_rows, L_cols
        
    def _compute_eigs(self, L_rows, L_cols):
        # extract eigenvectors of row and column graph laplacians
        eig_vals_row, eig_vecs_row = self._eigen(L_rows)
        eig_vals_col, eig_vecs_col = self._eigen(L_cols)
        
        return eig_vals_row, eig_vecs_row, eig_vals_col, eig_vecs_col
            
    def _eigen(self, A):
        eigenValues, eigenVectors = npla.eigh(A)
        idx = np.argsort(eigenValues)
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        return (eigenValues, eigenVectors)

    def _init_graph_basis(self, W):
        # gets basis returns laplacian
        W = W - np.diag(np.diag(W))
        D = np.diagflat(np.sum(W, 1))
        L = D - W
        return L

    def _squared_frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        return tensor_sum

        