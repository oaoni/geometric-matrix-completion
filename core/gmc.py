#Functional geometric matrix completion
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
import matplotlib
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.gridspec
from scipy.cluster.hierarchy import linkage
from tqdm.notebook import trange, tqdm

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

class GMC:
    "Functional geometric matrix completion"
    def __init__(self, name):
        self.name = name
        #Define the graph
        self.tf_graph = tf.Graph()



    def fit(self, num_iters, grid=[], plot=False, tune=False, cluster=True):



        if cluster:
            self.linkage = linkage(self.M, method='ward')

        g = self.tf_graph
        with g.as_default():

            train_loss_log = []
            test_loss_log = []
            train_corr_log = []
            test_corr_log = []
            iter_log = []

            if tune:
                IPython.display.clear_output()
                display(grid)
            best_test_corr = 0
            for iter in trange(num_iters, desc='Training w/ Params LR: {}, mu: {}, rho: {}, alpha: {}'.format(self.lr,
                               self.mu, self.rho, self.alpha)):

                if iter%50 == 0:
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

                    train_corr_np = np.corrcoef(train_ground,train_pred)[0][-1]
                    test_corr_np = np.corrcoef(test_ground,test_pred)[0][-1]
                    train_corr_log.append(train_corr_np)
                    test_corr_log.append(test_corr_np)

                    best_test_corr = np.array(test_corr_log).max()

                    #Record iteration
                    iter_log.append(iter)
                    if (plot) and (iter%5000 == 0):
                        IPython.display.clear_output()

                        print("iter " + str(iter) +" ,train loss: "+str(train_loss_np)+", test loss: " + str(test_loss_np)\
                               + " ,train corr: " + str(train_corr_np) + " ,test corr: " + str(test_corr_np)\
                               + " ,best test corr: ", str(best_test_corr))
                        print("Hyperparameters: lr: {}, m: {}, n: {}".format(self.lr, self.m, self.n))
                        X_np = self.sess.run(self.X)

                        if cluster:

                            #First create the clustermap figure
                            cg = sns.clustermap(X_np,row_linkage=self.linkage,col_linkage=self.linkage,figsize=(15,6))
                            # set the gridspec to only cover half of the figure
                            cg.gs.update(left=0.05, right=0.4)

                            #create new gridspec for the right part
                            gs2 = matplotlib.gridspec.GridSpec(1,1, left=0.45, right=0.7)
                            gs3 = matplotlib.gridspec.GridSpec(1,1, left=0.75, right=1.0)
                            # create axes within this new gridspec
                            ax2 = cg.fig.add_subplot(gs2[0])
                            ax3 = cg.fig.add_subplot(gs3[0])
                            # plot boxplot in the new axes
                            ax2.plot(iter_log[3:], train_loss_log[3:], 'r', iter_log[3:], test_loss_log[3:], 'b')
                            ax2.set_title("Train and Test Loss")
                            ax2.legend(['Train Loss','Test Loss'])

                            ax3.plot(iter_log[3:], train_corr_log[3:], 'r', iter_log[3:], test_corr_log[3:], 'b')
                            ax3.set_title("Train and Test Correlation")
                            ax3.legend(['Train Corr','Test Corr'])
                            plt.show()

                        else:
                            fig, ax = plt.subplots(1,4, figsize=(15,5))
                            ax[0].imshow(self.M)
                            ax[0].set_title("True")
                            ax[1].imshow(X_np)
                            ax[1].set_title("Training Reconstruction")
                            ax[2].plot(iter_log[3:], train_loss_log[3:], 'r', iter_log[3:], test_loss_log[3:], 'b')
                            ax[2].set_title("Train and Test Loss")
                            ax[2].legend(['Train Loss','Test Loss'])
                            ax[3].plot(iter_log[3:], train_corr_log[3:], 'r', iter_log[3:], test_corr_log[3:], 'b')
                            ax[3].set_title("Train and Test Correlation")
                            ax[3].legend(['Train Corr','Test Corr'])
                            plt.show()




                    #Make summary dic a model self method
                    summary_dic = {'learning_rate':self.lr,'m':self.m,'n':self.n,'mu':self.mu,'rho':self.rho,'alpha':self.alpha,
                                   'num_iters':num_iters,
                                   'train_loss':train_loss_log[-1],'test_loss':test_loss_log[-1],
                                   'best_train_loss':np.array(train_loss_log).min(),'best_test_loss':np.array(test_loss_log).min(),
                                   'train_corr':train_corr_log[-1],'test_corr':test_corr_log[-1],
                                   'best_train_corr':np.array(train_corr_log).max(),'best_test_corr':best_test_corr}

                    self.summary_dic = summary_dic

                self.sess.run(self.opt_op,feed_dict={'lr:0':self.lr})
                # self.sess.run(self.opt_op)

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
