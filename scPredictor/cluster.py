import numpy as np
import fastcluster as fc
from scipy.spatial import distance as scd
from scipy.cluster import hierarchy as shc

from sklearn.metrics.pairwise import pairwise_distances

def cluster_rows(X, metric='correlation', method='complete'):
    '''
    Cluster the rows of an array X

    Parameters
    ----------
    X : ndarray
        The rows of X will be clustered
    metric : str
        Metric to build pairwise distance matrix
    method : str
        Method to use for hierarchical clustering
        
    Returns
    -------
    list
        a list of leaf node ids
    '''
    c = pairwise_distances(X=X,metric=metric,n_jobs=-1) # create pairwise distance matrix
    np.fill_diagonal(c,0.) # make sure diagonal is not rounded
    c = scd.squareform(c,force='tovector',checks=False) # force distance matrix to vector
    c = np.nan_to_num(c) 
    z = fc.linkage(c, method=method) # create linkage
    z = z.clip(min=0)
    return shc.leaves_list(z)