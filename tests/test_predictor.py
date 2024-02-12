#from .context import predictor


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.insert(1, os.path.join(sys.path[0], '..'))

from predictor.predictor import Predictor
from predictor.io import load_from_seurat

#-----------------------------------------------------
# Test
#-----------------------------------------------------

from sklearn import datasets
import numpy as np
import pickle
print('done\n')

#CHOICE = 'iris'
CHOICE = 'seurat'

if CHOICE == 'iris':
    FOLDER = 'py_iris'
    iris = datasets.load_iris(return_X_y=False)
    M = iris.data
    features = iris.feature_names
    target = np.array([iris.target_names[x] for x in iris.target])
elif CHOICE == 'seurat':
    FOLDER = 'py_seurat'
    filepath = '/groups/irset/archives/web/UncoVer/studies/Young_2018_Science/scRNA-seq/Homo_sapiens/processed_data/subset/Young_2018_Science_processing_standard.subset.others_clin_ct.light.rds'
    M, target, features, coordinates, coord_labels = load_from_seurat(filepath, assay='RNA', slot='data', variablefeatures=True, metadata='CellType', coordinates='umap')
    
if CHOICE == 'iris':
    p = Predictor(M=M,
                  target=target,
                  features=features,
                  pct_train=.34,
                  transpose=False,
                  coordinates=M,
                  coordinate_labels=features)
elif CHOICE == 'seurat':
    p = Predictor(M=M,
                  target=target,
                  features=features,
                  pct_train=.3,
                  transpose=True,
                  coordinates=coordinates,
                  coordinate_labels=coord_labels)
print(p)

p.nn(
    hidden_layer_sizes=[128,10],
    solver='lbfgs',
    alpha=1e-5,
    random_state=1
)
print('DONE!\n')
print(p)

#param_grid = {
#    'hidden_layer_sizes' : [[3,10],[5,10], [10,10]],
#    'alpha' : [1e-5, 1e-3]
#}
#
#gs_params = {
#    'n_jobs':1,
#}
#
#p.nn(
#    key='nn_gs',
#    name='nn_gridsearch',
#    param_grid=param_grid,
#    gs_params=gs_params,
#    solver='lbfgs',
#    random_state=1
#)
#print('DONE!\n')
#print(p)

## Access all parameters of a given model
#print('nn_gs parameters:')
#print(p.get_model_params('nn_gs'))
#
## Look at GridSearchCV results
#print('NN GridSearchCV results:')
#print(p.get_cv_results('nn_gs', pandas=True))

param_grid = {
    'penalty' : [None, 'l1','l2','elasticnet'],
    'dual' : [True, False]
}

gs_params = {
    'n_jobs':-1,
}

p.logistic_regression(
    key='logreg_gs',
    name='logreg_gridsearch',
    param_grid=param_grid,
    gs_params=gs_params,
    max_iter=200
)
print('DONE!\n')
print(p)

#param_grid = {
#    'n_neighbors' : [3, 5, 7, 9, 13, 17],
#}
#
#gs_params = {
#    'n_jobs':1,
#}
#
#p.knn(
#    param_grid=param_grid,
#    gs_params=gs_params
#)
#print('DONE!\n')
#print(p)



# manual import
from sklearn.naive_bayes import GaussianNB

# generate model and specify argument values
gnb = GaussianNB(var_smoothing=1e-9)

# add model to Predictor object (training and validation will be conducted automatically)
p.train_sklearn_model(gnb,
                      key='gnb',
                      name='GaussianNaiveBayes')
print(p)


## manual import
#from sklearn.naive_bayes import GaussianNB
#
## generate model and specify argument values
#gnb = GaussianNB(var_smoothing=1e-9)
#
## set GridSearchCV parameter grid
#param_grid = {
#    'var_smoothing' : [1e-7, 1e-9, 1e-11]
#}
#
## run GridSearchCV and add model to Predictor object
#p.gridsearchcv(gnb,
#               key='gnb_gs',
#               name='GaussianNaiveBayes (gridsearch)',
#               param_grid=param_grid,
#               cv='split',
#               n_jobs=1)
#
#print(p)


p.plot_model_accuracy(savefig='../outputs/tests/model_acc.png', dpi=200)


p.delete('gnb')
print(p)

p.plot_proba_validation(key='nn',
                        interpolation='none',
                        onelimit=False,
                        cluster='both',
                        cmap='magma',
                        savefig='../outputs/tests/nn_validation.png',
                        dpi=140)
try:
    p.plot_proba(key='nn',
                 arr=M,
                 interpolation='none',
                 onelimit=False,
                 cluster='both',
                 savefig='../outputs/tests/proba.png')
except:
    p.plot_proba(key='nn',
                 arr=M.T,
                 interpolation='none',
                 onelimit=False,
                 cluster='both',
                 savefig='../outputs/tests/proba.png')

print('get predictions')
try:
    predictions = p.predict(key='nn', arr=M) # get prediction labels (n_samples)
except:
    predictions = p.predict(key='nn', arr=M.T) # get prediction labels (n_samples)

print('get prediction probabilities')
try:
    predictions_proba = p.predict_proba(key='nn', arr=M) # get prediction probabilities (n_samples, n_features)
except:
    predictions_proba = p.predict_proba(key='nn', arr=M.T) # get prediction probabilities 

p.scatter_validation(model='logreg_gs',
                    x=0,
                    y=1,
                    gridsize=18,
                    cmap='YlOrRd',
                    savefig='../outputs/tests/scatter_validation.png')

p.print_report('nn')

p.print_report_help('nn')

p.print_coef('logreg_gs')

coef = p.get_coef('logreg_gs')

try:
    p.scatter_feature(feature='petal width (cm)', x=0, y=1, savefig='../outputs/tests/scatter_feature.png', dpi=140)
except:
    p.scatter_feature(feature='HBG1', x=0, y=1, savefig='../outputs/tests/scatter_feature.png', dpi=140)


p.save('../outputs/tests/pred_obj')
with open('../outputs/tests/pred_obj', 'rb') as f:
    p2 = pickle.load(f)
print(p2)

# Save logistic regression model
p.save_model('logreg_gs', '../outputs/tests/logreg.pkl')

# Load logistic regression model from file
with open('../outputs/tests/logreg.pkl','rb') as f:
    logreg = pickle.load(f)
    
print(logreg)