#data processing functions
import h5py
import scipy.sparse as sp
import numpy as np
import pandas as pd
from numpy import linalg as npla
import os
import random
import scipy
import umap
import umap.plot
from umap import UMAP
from sklearn.cluster import KMeans
import networkx as nx

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds[ 'ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out

def load_gi_example(frac):
    """
    loads gi delta data
    """
    path_dataset = 'GeometricMatrixCompletion/data/gi_delta.csv'
    side_dataset = 'GeometricMatrixCompletion/data/gi_delta_side.csv'

    M = pd.read_csv(path_dataset,index_col=0)
    side = pd.read_csv(side_dataset,index_col=0)

    #Two functions for sampling from the phenotype matrix
    random.seed(30)

    #Sample a percentage of the genes
    M_train, M_test, S_train, S_test = sample_mask(M,frac,use_upper=True)
    M_train = M_train.values
    M_test = M_test.values
    S_train = S_train.values
    S_test = S_test.values

    return [M,S_train,S_test,M_train,M_test],side

def gi_side_process(side,form,plot=True):
    """ Various forms of processing side information for gi side information
    """

    if form == 'corr':
        processed_side = scipy.linalg.sqrtm(side.T.corr())

    elif form == 'raw_sim_set':
        side = side.values
        processed_side = umap_graph(side, plot=plot)
    elif form == 'corr_sim_set':
        side = side.T.corr().values
        processed_side = umap_graph(side, plot=plot)
    elif form == 'sqrt_sim_set':
        side = scipy.linalg.sqrtm(side.T.corr())
        processed_side = umap_graph(side, plot=plot)


    laplacian = graph_laplacian(processed_side)
    W_val,W_vec = eigen(laplacian)


    return W_val,W_vec

def umap_graph(data, plot=True):
    near_n = 15
    min_d = 0.1
    rand_s = np.random.RandomState(30)

    transformer = UMAP(n_neighbors=near_n,random_state=rand_s,min_dist=min_d).fit(data)
    raw_emb = transformer.transform(data);
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=rand_s)
    labels = kmeans.fit(raw_emb)

    node_l = labels.labels_

    sim_set = umap.umap_.fuzzy_simplicial_set(data,near_n,random_state=rand_s,\
                                              metric="euclidean")
    sim_set = sim_set[0].toarray()
    if plot:
        nx.draw(nx.DiGraph(sim_set), arrows=False, node_size=30, width=0.1, node_color = node_l, cmap='viridis')

    return sim_set

def load_example(dataset='netflix'):
    """
    loads synthetic netflix (netflix) or flixster datasets
    """

    if dataset == 'netflix':
        path_dataset = './GeometricMatrixCompletion/data/synthetic_netflix.mat'

        #Data
        M = load_matlab_file(path_dataset, 'M')
        S_train = load_matlab_file(path_dataset, 'Otraining')
        S_test = load_matlab_file(path_dataset, 'Otest')
        M_train = np.array(M)*np.array(S_train)
        M_test = np.array(M)*np.array(S_test)

        #Side information
        W_rows = load_matlab_file(path_dataset, 'Wrow').todense()  # Row Graph
        W_cols = load_matlab_file(path_dataset, 'Wcol').todense()  # Column Graph

    elif dataset == 'flixster':
        path_dataset = './GeometricMatrixCompletion/data/training_test_dataset_10_NNs.mat'

        #Data
        M = load_matlab_file(path_dataset, 'M')
        S_train = load_matlab_file(path_dataset, 'Otraining')
        S_test = load_matlab_file(path_dataset, 'Otest')
        M_train = np.array(M)*np.array(S_train)
        M_test = np.array(M)*np.array(S_test)

        #Side information
        W_rows = load_matlab_file(path_dataset, 'W_users') # Row Graph
        W_cols = load_matlab_file(path_dataset, 'W_movies') # Column Graph

    # extract Laplacians of the row and column graphs
    eig_vals_row, eig_vecs_row = init_graph_basis(W_rows)
    eig_vals_col, eig_vecs_col = init_graph_basis(W_cols)


    return [M,S_train,S_test,M_train,M_test],[W_rows,W_cols],[eig_vals_row,eig_vecs_row,eig_vals_col,eig_vecs_col]

def eigen(A):
    eigenValues, eigenVectors = npla.eigh(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return (eigenValues, eigenVectors)

def init_graph_basis(W):
    # gets basis returns eig_vals and eig_vecs
    W = W - np.diag(np.diag(W))
    D = np.diagflat(np.sum(W, 1))
    L = D - W
    eig_vals, eig_vecs = eigen(L)
    return eig_vals, eig_vecs

def upper_triangle(M, k=1):
    """Return the upper triangular part of a matrix in stacked format (i.e. as a vector)
    """
    keep = np.triu(np.ones(M.shape), k=k).astype('bool').reshape(M.size)
    # have to use dropna=FALSE on stack, otherwise will secretly drop nans and upper triangle
    # will not behave as expected!!
    return M.stack(dropna=False).loc[keep]


def sample_mask(M,frac,use_upper=False,use_index=False,random_state=np.random.RandomState(30)):
    """ Samples a fraction of marix elements
    returns: train/test matrix (M), and Mask (S)
    """

    if use_upper:
        #Samples from upper triangular. For symmetrical relational matrices.
        multInd = upper_triangle(M, k=0).sample(frac=frac, replace=False,
                                               random_state=random_state).index

        #Initialize indicators
        S_train = pd.DataFrame(0, index=M.index, columns=M.columns)
        S_test = pd.DataFrame(1, index=M.index, columns=M.columns)

        for Ind in multInd:
            S_train.loc[Ind] = 1
            S_test.loc[Ind] = 0

            S_train.loc[Ind[::-1]] = 1
            S_test.loc[Ind[::-1]] = 0


        M_train = M*S_train
        M_test = M*S_test

    elif use_index:
        #Samples from entire matrix
        #Produce multiindex of sampled elements (might want to return)
        multInd = M.stack(dropna=False).sample(frac=frac, replace=False,
                                              random_state=random_state).index

        #Initialize indicators
        S_train = pd.DataFrame(0, index=M.index, columns=M.columns)
        S_test = pd.DataFrame(1, index=M.index, columns=M.columns)

        for Ind in multInd:
            S_train.loc[Ind] = 1
            S_test.loc[Ind] = 0

        M_train = M*S_train
        M_test = M*S_test
    else:
        test_frac = 1-frac
        m,n = M.shape
        S = np.random.rand(m,n)
        S_train = (S < frac)
        S_test = (S >= (1-test_frac))
        M_train = np.multiply(M,S_train)
        M_test = np.multiply(M,S_test)

    return M_train, M_test, S_train, S_test

def graph_laplacian(W):
    # returns graph laplacian
    W = W - np.diag(np.diag(W))
    D = np.diagflat(np.sum(W, 1))
    L = D - W

    return L
