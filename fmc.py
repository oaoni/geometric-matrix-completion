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

class FMC:
    "Functional geometric matrix completion"
    def __init__(self,M,M_training,M_test,S_training,S_test,
                 W_rows,W_cols,p_max,q_max,p_init,q_init,lr,m,n,use_side=True,cluster=True):



        self.lr = lr
        self.m = m
        self.n = n
        self.M = M
        self.S_train = S_training
        self.S_test = S_test
        self.cluster = cluster
        self.linkage = linkage(M, method='ward')

        if use_side:
            L_rows, L_cols = self._compute_laplacians(W_rows, W_cols)
            eig_vals_row, eig_vecs_row, eig_vals_col, eig_vecs_col = self._compute_eigs(L_rows, L_cols, m, n)
        else:
            eig_vecs_row = np.zeros((M.shape[0], m))
            eig_vecs_col = np.zeros((M.shape[1], n))
            eig_vals_row = np.zeros(m)
            eig_vals_col = np.zeros(n)

        #Define the graph
        self.tf_graph = tf.Graph()
        g = self.tf_graph
        with g.as_default():
            #Build model
            self.build_model(M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                        eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n)




    def build_model(self,M_training,M_test,S_training,S_test,eig_vecs_row,eig_vecs_col,
                    eig_vals_row,eig_vals_col,p_max,q_max,p_init,q_init,m,n):
        """Create graph"""

        # tf.reset_default_graph()

        lr = tf.placeholder(tf.float32, name="lr")
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

        E_tot = E_data + 0.00001*E_comm
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.opt_op = optimizer.minimize(E_tot)

        #Metrics
        self.train_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_training_tf) - M_training_tf)/ tf.reduce_sum(S_training_tf))
        #validation_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(S_validation_tf, (X - M))) / tf.reduce_sum(S_validation_tf))
        self.test_loss = tf.sqrt(self._squared_frobenius_norm(tf.multiply(self.X, S_test_tf)- M_test_tf)/tf.reduce_sum(S_test_tf))

        ground = tf.placeholder(tf.float32, name='ground')
        mask = tf.placeholder(tf.float32, name='mask')

        self.corr_metric = self._corr_metric(ground=ground, pred=self.X, mask=mask)

    def fit(self, num_iters, grid=[], plot=False):
        g = self.tf_graph
        with g.as_default():
            # Create a session for running Ops on the Graph.
            self.config = tf.ConfigProto(allow_soft_placement=True)
            self.config.gpu_options.allow_growth = True
            # self.sess = tf.Session(config=self.config)
            self.sess = tf.Session(config=self.config)

            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())

            train_loss_log = []
            test_loss_log = []
            train_corr_log = []
            test_corr_log = []
            iter_log = []

            for iter in range(num_iters):

                if iter%1000 == 0:
                    #Compute training and test loss
                    train_loss_np, test_loss_np = self.sess.run([self.train_loss, self.test_loss])
                    train_loss_log.append(train_loss_np)
                    test_loss_log.append(test_loss_np)

                    #Produce training and test correlation vectors
                    train_corr = self.sess.run(self.corr_metric,feed_dict={'ground:0':self.M, 'mask:0':self.S_train})
                    test_corr = self.sess.run(self.corr_metric,feed_dict={'ground:0':self.M, 'mask:0':self.S_test})
                    #Compute correlation
                    train_ground, train_pred = train_corr
                    test_ground, test_pred = test_corr

                    train_corr_log.append(np.corrcoef(train_ground,train_pred)[0][-1])
                    test_corr_log.append(np.corrcoef(test_ground,test_pred)[0][-1])

                    #Record iteration
                    iter_log.append(iter)
                    if (plot) and (iter%5000 == 0):
                        IPython.display.clear_output()

                        print("iter " + str(iter) +" ,train loss: "+str(train_loss_np)+", test loss: " + str(test_loss_np) )
                        print("Hyperparameters: lr: {}, m: {}, n: {}".format(self.lr, self.m, self.n))
                        X_np = self.sess.run(self.X)
                        fig, ax = plt.subplots(1,2, figsize=(15,5))
                        if self.cluster:

                            cg1 = sns.clustermap(self.M,
                                row_linkage=self.linkage,
                                col_linkage=self.linkage,figsize=(6,6))

                            cg2 = sns.clustermap(X_np,
                                row_linkage=self.linkage,
                                col_linkage=self.linkage,figsize=(6,6))

                        else:
                            ax[0].imshow(self.M)
                            ax[0].set_title("True")
                            ax[1].imshow(X_np)
                            ax[1].set_title("X")
                        ax[0].plot(iter_log[3:], train_loss_log[3:], 'r', iter_log[3:], test_loss_log[3:], 'b')
                        ax[0].set_title("Train and Test Loss")
                        ax[1].plot(iter_log[3:], train_corr_log[3:], 'r', iter_log[3:], test_corr_log[3:], 'b')
                        ax[1].set_title("Train and Test Correlation")
                        plt.legend(['Train Loss','Test Loss'])
                        plt.show()
                        display(grid)



                    summary_dic = {'learning_rate':self.lr,'m':self.m,'n':self.n,'num_iters':num_iters,
                                   'train_loss':train_loss_log[-1],'test_loss':test_loss_log[-1],
                                   'best_train_loss':np.array(train_loss_log).min(),'best_test_loss':np.array(test_loss_log).min(),
                                   'train_corr':train_corr_log[-1],'test_corr':test_corr_log[-1],
                                   'best_train_corr':np.array(train_corr_log).max(),'best_test_corr':np.array(test_corr_log).max()}

                    self.summary_dic = summary_dic

                self.sess.run(self.opt_op,feed_dict={'lr:0':self.lr})
                # self.sess.run(self.opt_op)

    def _compute_laplacians(self, W_rows, W_cols):
        # produce Laplacians of the row and column graphs
        L_rows = self._init_graph_basis(W_rows)
        L_cols = self._init_graph_basis(W_cols)

        return L_rows, L_cols

    def _compute_eigs(self, L_rows, L_cols, m, n):
        # extract eigenvectors of row and column graph laplacians
        eig_vals_row, eig_vecs_row = self._eigen(L_rows)
        eig_vals_col, eig_vecs_col = self._eigen(L_cols)

        return eig_vals_row[0:m], eig_vecs_row[:,0:m], eig_vals_col[0:n], eig_vecs_col[:,0:n]

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

    def _corr_metric(self, ground, pred, mask):

        flat_mask = tf.reshape(mask,[-1]) #flatten tensor
        flat_mask = tf.cast(flat_mask, tf.bool) #astype boolean mask
        ground_linear = tf.cast(tf.boolean_mask(tf.reshape(ground,[-1]), flat_mask),tf.float32)
        pred_linear = tf.cast(tf.boolean_mask(tf.reshape(pred,[-1]), flat_mask), tf.float32)

        # corr = tf.contrib.metrics.streaming_pearson_correlation(ground_linear,pred_linear)

        return ground_linear, pred_linear

    def _to_numpy(self,arrays):

        arrays_ = [array.values if isinstance(array, pd.DataFrame) else array for array in arrays]

        return arrays_
