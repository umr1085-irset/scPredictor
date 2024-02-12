import numpy as np
import scipy.sparse as ss

def read_M_from_seurat(obj, assay, slot, variablefeatures):
    '''
    Read matrix and feature labels from Seurat file
    
    Parameters
    ----------
    obj : obj
        Seurat object read with rpy2
    assay : str
        Name of assay
    slot : str
        Name of slot
    variablefeatures : bool
        Variable features only or not
        
    Returns
    -------
    sparse matrix
        data matrix from Seurat file
    '''
    # dive into RDS object
    assays = obj.slots['assays']
    try:
        assay = assays.rx2(assay)
    except:
        raise Exception('Provided assay name does not exist')
    try: 
        matrix = assay.slots[slot]
    except:
        raise Exception('Provided slot name does not exist')
    
    i = np.array(matrix.slots['i']) # get row indices
    p = np.array(matrix.slots['p']) # get column pointer indices
    x = np.array(matrix.slots['x']) # get values
    shape = tuple(matrix.slots['Dim']) # get shape
    M = ss.csc_array((x, i, p), shape=shape) # build sparse matrix
    features = np.array(matrix.slots['Dimnames'][0]) # extract features
    
    if variablefeatures: # if True
        varfeat = np.array(assay.slots['var.features']) # get variable features
        idx = np.array([np.where(features==vf)[0][0] for vf in varfeat]) # get variable feature indices
        return M[idx,:], varfeat # return sliced matrix, variable features
    else: # if False
        return M, features # return matrix, features

def read_metadata_from_seurat(obj, metadata):
    '''
    Read metadata labels from Seurat file
    
    Parameters
    ----------
    obj : robj
        Seurat object read with rpy2
    metadata : str
        Name of target metadata
        
    Returns
    -------
    array
        labels from Seurat metadata
    '''
    df = obj.slots['meta.data']
    return np.array(df.rx2(metadata)) # return array of values

def read_coordinates_from_seurat(obj, coordinates):
    '''
    Read cell embeddings and labels hahafrom Seurat file
    
    Parameters
    ----------
    obj : robj
        Seurat object read with rpy2
    coordinates : str
        Name of coordinates
        
    Returns
    -------
    matrix
        Coordinates
    '''
    red = obj.slots['reductions'].rx2(coordinates) # get reduction slot
    coordinates = np.array(red.slots['cell.embeddings'].rx()) # get values
    lbl_key = np.array(red.slots['key'].rx())[0] # get reduction key
    coordinate_labels = [f'{lbl_key}{i}' for i in range(1,coordinates.shape[1]+1)] # create coordinate labels
    return coordinates, coordinate_labels # return coordinate values, coordinate labels
    
def load_from_seurat(filepath, assay='', slot='data', variablefeatures=False, metadata='', coordinates=''):
    '''
    Load data matrix (features x items) and target labels from Seurat RDS file
    
    Parameters
    ----------
    filepath : str
        Path to Seurat RDS file
    assay : str
        Name of assay
    slot : str
        Name of slot. Defaults to data
    metadata : str
        Name of target metadata
    coordinates : str
        Name of projection
    
    Returns
    -------
    sparse matrix
        data matrix from Seurat file
    array
        labels from Seurat metadata
    array
        projection coordinates
    array
        projection coordinate labels
    '''
    from rpy2.robjects.packages import importr
    
    base = importr("base") # base R
    obj = base.readRDS(filepath) # read RDS file
    M, features = read_M_from_seurat(obj, assay, slot, variablefeatures) # get matrix, feature labels
    target = read_metadata_from_seurat(obj, metadata) # get metadata values
    
    if coordinates: # if True
        coordinates, coordinate_labels = read_coordinates_from_seurat(obj, coordinates) # get coordinate values, coordinate labels
        return M, target, features, coordinates, coordinate_labels # return
    else:
        return M, target, features # return