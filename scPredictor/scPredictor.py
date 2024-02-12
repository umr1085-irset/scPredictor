import numpy as np
import matplotlib
from matplotlib import pyplot as plt

class Predictor:
    '''
    Predictor class
    
    Attributes
    ----------
    M : ndarray or sparse matrix
        Data matrix. Must have variables as columns, see constructor function's transpose parameter
    target : list or array
        List of target labels
    features : list or array or None
        Feature names. Defaults to None
    shape : tuple
        Data shape
    models : dict
        Dictionary to store models (keys are model identifiers)
    pct_train : float
        Training percentage to split training and validation data. Must be between 0 and 1
    coordinates : matrix
        Coordinate values. Optional
        
    Methods
    -------
    set_pct_train(pct_train=.8)
        Set training percentage
    set_coordinates(x, y)
        Set coordinates
    get_train_val_indices()
        Get training and validation indices
    get_split_data()
        Get data and target labels for both training and validation data
    get_training_data()
        Get training data and target labels
    get_validation_data()
        Get validation data and target labels
    get_model(key)
        Get given model
    get_coordinates()
        Get coordinates
    print_report(key)
        Print classification report for given model
    print_report_help(key)
        Print report help and classification report for given model
    get_report(key)
        Get classification report for given model
    print_coef(key)
        Print model coefficients if available, for given model
    get_coef(key)
        Get model coefficients for given model
    add_model(key, model)
        Add model to collection
    available_models()
        Get list of available models
    predict(model, arr)
        Predict target values for given data from given model
    predict_proba(model, arr)
        Get probability values of predictions for given data from given model
    plot_proba(model, arr, **kwargs)
        Plots a heatmap of prediction probabilities for each element of array
    plot_proba_validation(model, **kwargs)
        Plots a heatmap of prediction probabilities with elements split by target for given model
    plot_model_accuracy(savefig='', dpi=200)
        Plots a barchart of the models accuracies
    train_model(skmodel, key, name='Sklearn model')
        Train and store ML model
    logistic_regression(name='Logistic Regression', key='logreg', **kwargs)
        Build a Logistic Regression model
    knn(name='K-Nearest Neighbors', key='knn', **kwargs)
        Build a K-Nearest Neighbors model
    nn(name='Neural Network', key='nn', layers=[128], epochs=10)
        Build a Fully Connected Neural Network
    scatter_validation(model='', bgcolor='lightgrey', bgsize=1, bgalpha=1, alpha=1, \
                       dotsize=5, linewidth=0, removeticks=True, colormap='tab10', \
                       lgdloc='upper center', lgdncol=4, lgdbbox_to_anchor=(0.5, 0.3), \
                       lgdframeon=True, savefig='', dpi=200)
        Plots scatter plots of training and validation data, with true and predicted labels
    delete(key='')
        Delete a given model
    save(file)
        Save Predictor object
    save_model(key, file)
        Save
    '''
    
    def __init__(self, M, target, features=None, pct_train=.8, transpose=False, coordinates=None, coordinate_labels=None):
        '''
        Parameters
        ----------
        M : array or sparse matrix
            Data
        target : array
            Target labels
        features : array. Optional
            Feature names
        pct_train : float
            Percentage of data used as training data
        Transpose : bool
            Whether to transpose the data matrix or not. Features should be columns
        coordinates : matrix
            Coordinate values. Optional
        coordinate_labels : array
            Coordinate labels. Optional unless coordinates are provided
        '''
        if coordinates is not None and coordinate_labels is None:
            raise Exception('If coordinates are provided, coordinate_labels are required. Please use the coordinate_labels parameter')
            
        if transpose:
            self.M = M.T
        else:
            self.M = M
        
        self.models = {}
        self.target = np.array(target)
        self.features = np.array(features)
        self.shape = self.M.shape
        self.set_pct_train(pct_train)
        self.set_coordinates(coordinates)
        self.coordinate_labels = np.array(coordinate_labels)
        
    def set_pct_train(self, pct_train=.8):
        '''
        Set training percentage value and pick training and validation indices
        
        Parameters
        ----------
        pct_train : float
            Percentage of data used as training data
        '''
        if pct_train <=0 or pct_train >= 1:
            raise Exception('Training percentage must be above 0 and below 1')
        self.pct_train = pct_train
        m = self.M.shape[0] # number of samples
        n = int(m*pct_train) # number of training samples
        self.train_idx = np.sort(np.random.choice(m, n, replace=False)) # get n random sample indices
        self.val_idx = np.setdiff1d(range(m), self.train_idx) # get the validation set indices
        test_fold = np.repeat(-1, m) # create an array of -1 values
        test_fold[self.val_idx] = 0 # set values to 0 for validation indices
        self.test_fold = test_fold
        
    def set_coordinates(self, coordinates):
        '''
        Set coordinates
        
        Parameters
        ----------
        coordinates : matrix
            Coordinate values
        '''
        self.coordinates = coordinates
        
    def set_coordinate_labels(self, coordinates):
        '''
        Set coordinates
        
        Parameters
        ----------
        coordinates : matrix
            Coordinate values
        '''
        self.coordinate_labels = np.array(coordinate_labels)
        
    def get_train_val_indices(self):
        '''
        Get training and validation indices
        
        Returns
        -------
        array
            Training indices
        array
            Validation indices
        '''
        return self.train_idx, self.val_idx
        
    def get_split_data(self):
        '''
        Get data and target labels for both training and validation data
        
        Returns
        -------
        matrix
            Training data
        matrix
            Validation data
        array
            Training target labels
        array
            Training validation labels
        '''
        return self.M[self.train_idx,:], self.M[self.val_idx,:], self.target[self.train_idx], self.target[self.val_idx]
    
    def get_training_data(self):
        '''
        Get training data and target labels
        
        Returns
        -------
        matrix
            Training data
        array
            Training target labels
        '''
        return self.M[self.train_idx,:], self.target[self.train_idx]
    
    def get_validation_data(self):
        '''
        Get validation data and target labels
        
        Returns
        -------
        matrix
            Validation data
        array
            Validation target labels
        '''
        return self.M[self.val_idx,:], self.target[self.val_idx]
        
    def get_model(self, key):
        '''
        Get given model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        
        Returns
        -------
        model
            Requested model
        '''
        return self.models[key]
    
    def get_coordinates(self):
        '''
        Get coordinates

        Returns
        -------
        matrix
            Coordinates values
        '''
        return self.coordinates
    
    def get_coordinates_by_index(self, i):
        '''
        Get ith coordinates column
        
        Parameters
        ----------
        i : int
            Column index
            
        Returns
        -------
        array
            Array of coordinates
        '''
        return self.coordinates[:,i]
    
    def get_M_feature(self, feature):
        '''
        Get matrix row values for a given feature
        
        Parameters
        ----------
        feature : str
            Name of feature
            
        Returns
        -------
        array
            Matrix row
        '''
        try:
            i = np.where(self.features == feature)[0][0]
        except:
            raise Exception(f'Feature: {feature} not found in features names')
        try:
            array = self.M[:,[i]].toarray().flatten()
        except:
            array = self.M[:,[i]].flatten()
        return array
    
    def print_report(self, key):
        '''
        Print classification report for given model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        '''
        print(self.models[key].report)

    def print_report_help(self, key):
        '''
        Print report help and classification report for given model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        '''
        print('Precision: ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0.\n')
        print('Recall: ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples. The best value is 1 and the worst value is 0.\n')
        print('F1 score: can be interpreted as a harmonic mean of the precision and recall, F1 = 2 * (precision * recall) / (precision + recall). The best value is 1 and the worst value is 0.\n')
        print('Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.\n\n')
        print(self.models[key].report)
        
    def get_report(self, key):
        '''
        Get classification report for given model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        
        Returns
        -------
        dict
            Classification report
        '''
        return self.models[key].report_dict
    
    def print_coef(self, key):
        '''
        Print model coefficients if available, for given model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        '''
        import pandas as pd

        try:
            coef = self.models[key].model.coef_
            if coef.shape[0] == 1:
                try:
                    print(pd.Series(coef.flatten(), index=self.features))
                except:
                    print(pd.Series(coef, index=self.features))
            else:
                print(pd.DataFrame(coef, columns=self.features, index=self.models[key].class_names))
        except:
            raise Exception(f"Can't retrieve coef_ attribute for this model: {key}")
            
    def get_coef(self, key):
        '''
        Get model coefficients for given model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        
        Returns
        -------
        array
            Model coefficients
        '''
        try:
            return self.models[key].model.coef_ # sklearn standard
        except:
            raise Exception(f"Can't retrieve coef_ attribute for this model: {key}")
            
    def get_model_params(self, key):
        '''
        Get all model parameters
        
        Parameters
        ----------
        key : str
            Unique model identifier
            
        Returns
        -------
        dict
            Model parameters
        '''
        return self.models[key].model.get_params()
            
    def get_cv_results(self, key, pandas=False):
        '''
        Get cross validation results for given model
        
        Parameters
        ----------
        key : str
            Model identifier
        pandas : bool
            Return result as a Pandas DataFrame
        
        Returns
        -------
        dict or DataFrame
            Cross validation results
        '''
        import pandas as pd

        if pandas:
            return pd.DataFrame(self.models[key].cv_results)
        else:
            return self.models[key].cv_results 
    
    def get_cv_best_params(self, key):
        '''
        Get cross validation best parameters
        
        Parameters
        ----------
        key : str
            Model identifier
            
        Returns
        dict
            Best parameter values
        '''
        return self.models[key].cv_best_params
        
    
    def add_model(self, key, model):
        '''
        Add model to collection
        
        Parameters
        ----------
        key : str
            Model identifier
        model : model
            Model to store
        '''
        self.models[key] = model
        
    def available_models(self):
        '''
        Get list of available models
        
        Returns
        -------
        list
            Available models in Predictor object
        '''
        return list(self.models.keys())
    
    def predict(self, key, arr):
        '''
        Predict target values for given data from given model
        
        Parameters
        ----------
        key : str
            Model identifier
        arr : matrix
            Data matrix to generate predictions
            
        Returns
        -------
        array
            Predictions for data
        '''
        return self.models[key].predict(arr)
    
    def predict_proba(self, key, arr):
        '''
        Get probability values of predictions for given data from given model
        
        Parameters
        ----------
        key : str
            Model identifier
        arr : matrix
            Data matrix to generate predictions
            
        Returns
        -------
        array
            Prediction probabilities
        '''
        return self.models[key].predict_proba(arr)
        
    def plot_proba(self, key, arr, cmap='magma', interpolation='none', onelimit=False, cluster='none', savefig='', dpi=200, **kwargs):
        '''
        Plot a heatmap of prediction probabilities for each element of array, given a model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        arr : 2D array
            Data to predict
        cmap : str
            Colormap for heatmap
        interpolation : str
            Heatmap interpolation value. See Matplotlib's imshow() documentation
        onelimit : bool
            Set the probability scale max to 1
        cluster : str
            Clustering options. Must be one of none, rows, cols, both
        savefig : str
            Path to save figure
        dpi : int
            Figure's dots per inch. Defaults to 200
        '''
        from .cluster import cluster_rows

        predictions = self.models[key].predict_proba(arr)
        xlabels = self.models[key].class_names
        x = range(len(xlabels))
        
        if onelimit:
            clim_max = 1
        else:
            clim_max = predictions.max()
            
        if cluster == 'rows':
            ridx = cluster_rows(predictions, **kwargs)
            predictions = predictions[ridx,:]
        elif cluster == 'cols':
            cidx = cluster_rows(predictions.T, **kwargs)
            predictions = predictions[:,cidx]
            xlabels = xlabels[cidx]
        elif cluster == 'both':
            ridx = cluster_rows(predictions, **kwargs)
            predictions = predictions[ridx,:]
            cidx = cluster_rows(predictions.T, **kwargs)
            predictions = predictions[:,cidx]
            xlabels = xlabels[cidx]
            
        im = plt.imshow(predictions, aspect='auto', cmap=cmap, interpolation=interpolation)
        im.set_clim(0,clim_max)
        plt.colorbar()
        plt.xticks(x,xlabels,rotation=90)
        plt.title(f'Prediction probabilities\n{self.models[key].name}')
        plt.ylabel('Input samples')
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig,dpi=dpi)
        plt.close()
        
    def plot_proba_validation(self, key, cmap='magma', interpolation='none', onelimit=False, cluster='none', savefig='', dpi=200, **kwargs):
        '''
        Plot a heatmap of prediction probabilities split by target (validation data and labels)
        
        Parameters
        ----------
        key : str
            Model unique identifier
        cmap : str
            Colormap for heatmap
        interpolation : str
            Heatmap interpolation value. See Matplotlib's imshow() documentation
        onelimit : bool
            Set the probability scale max to 1
        cluster : str
            Clustering options. Must be one of none, rows, cols, both
        savefig : str
            Path to save figure
        dpi : int
            Figure's dots per inch. Defaults to 200
        '''
        from .cluster import cluster_rows

        X_val, lbls_val = self.get_validation_data()
        predictions = self.models[key].predict_proba(X_val)
        unique_val_labels = np.unique(lbls_val)
        xlabels = self.models[key].class_names
        x = range(len(xlabels))
        
        heights = []
        for i,lbl in enumerate(unique_val_labels):
            idx = np.where(lbls_val == lbl)[0]
            heights.append(len(idx))
        
        if onelimit:
            clim_max = 1
        else:
            clim_max = predictions.max()

        fig = plt.figure(constrained_layout=True)
        subfig = fig.subfigures(nrows=1, ncols=1)
        subfig.suptitle(f'Prediction probabilities of validation data\n{self.models[key].name} (key: "{key}")')
        subfig.supxlabel('Predictions')
        subfig.supylabel('Validation data')
        axs = subfig.subplots(nrows=len(unique_val_labels), ncols=1, sharex=True, gridspec_kw={'height_ratios': heights})
        
        if cluster=='both' or cluster=='cols':
            cidx = cluster_rows(predictions.T, **kwargs)
            predictions = predictions[:,cidx]
            xlabels = xlabels[cidx]
        
        for i,lbl in enumerate(unique_val_labels):
            idx = np.where(lbls_val == lbl)[0]
            sub = predictions[idx,:]
            
            if cluster == 'both' or cluster=='rows':
                ridx = cluster_rows(sub, **kwargs)
                sub = sub[ridx,:]
            
            y = [len(idx)/2]
            im = axs[i].imshow(sub, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation=interpolation)
            im.set_clim(0,clim_max)
            axs[i].set_yticks(y, [lbl])
            axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        axs[i].set_xticks(x, xlabels, rotation=90)
        subfig.colorbar(im, ax=axs)

        if savefig:
            plt.savefig(savefig,dpi=dpi)
        plt.close()
        
    def plot_model_accuracy(self, savefig='', dpi=200):
        '''
        Plot a barchart of the models accuracies
        
        Parameters
        ----------
        savefig : str
            Path to figure
        dpi : int
            Dots per inch
        '''
        names = []
        acc = []
        for k in self.models:
            names.append(f'{self.models[k].key} ({self.models[k].name})')
            acc.append(self.models[k].accuracy*100)
        idx = np.argsort(acc)
        names = np.array(names)[idx]
        acc = np.array(acc)[idx]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = range(len(acc))
        plt.bar(x, acc, color='#4682B4')
        plt.xticks(x, names, rotation=45, ha='right')
        plt.ylim(0,100)
        plt.title("Model accuracies using validation data")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel('Accuracy (%)')
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig,dpi=dpi)
        plt.close()
        
    def post_training(self, key, name, model, X_val, lbls_val, cv_results=None, cv_best_params=None):
        from sklearn.metrics import classification_report
        from .model import Model

        try:
            acc = model.score(X_val, lbls_val) # get score
        except:
            acc = model.score(X_val.toarray(), lbls_val)
        print(f'Validating model: {acc*100:.2f}% accuracy using validation set\n')
        try:
            lbls_pred = model.predict(X_val) # get predicted labels for validation data
        except:
            lbls_pred = model.predict(X_val.toarray())
        report = classification_report(lbls_val, lbls_pred, target_names=model.classes_.astype(str)) # get classification report
        report_dict = classification_report(lbls_val, lbls_pred, target_names=model.classes_.astype(str), output_dict=True) # get classification report as dictionary
        sklearnmodel = Model(key, name, model, acc, model.classes_, report, report_dict, cv_results=cv_results, cv_best_params=cv_best_params) # create Model (class instance)
        self.add_model(key, sklearnmodel) # store model in Predictor object
        
    def train_sklearn_model(self, skmodel, key, name='Sklearn model', forcearray=False):
        '''
        Train and store ML model
        
        Parameters
        ----------
        skmodel : Scikit-learn model
            An instanciated Scikit-learn model
        key : str
            Unique model identifier
        name : str
            Model name
        '''
        print('Getting training & validation sets')
        X_train, X_val, lbls_train, lbls_val = self.get_split_data() # get training data and labels, validation data and labels
        try:
            model = skmodel.fit(X_train, lbls_train) # train model
        except:
            model = skmodel.fit(X_train.toarray(), lbls_train) # train model
        
        self.post_training(key, name, model, X_val, lbls_val)

    def gridsearchcv(self, model, key, name, param_grid, cv='split', **kwargs):
        '''
        Run GridSearchCV for a given model

        Parameters
        ----------
        key : str
            Unique model identifier
        name : str
            Model name
        param_grid : dict
            Dictionary of parameters for grid search. Optional
        cv : str
            Cross validation option. Must be "split", int, cross-validation generator, an iterable or None
        '''
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import PredefinedSplit
        from .utils import printd

        if cv=='split':
            cv = PredefinedSplit(self.test_fold)

        print('Running GridSearchCV')
        clf = GridSearchCV(model, param_grid, cv=cv, **kwargs)
        try:
            clf.fit(self.M, self.target)
        except:
            clf.fit(self.M.toarray(), self.target)

        print('Retrieving best model')
        model = clf.best_estimator_
        print("Best model's parameters are:")
        printd(clf.best_params_, indent=1)

        X_val, lbls_val =self.get_validation_data() # get validation data and labels
        self.post_training(key, name, model, X_val, lbls_val, cv_results=clf.cv_results_, cv_best_params=clf.best_params_)

    def logistic_regression(self, key='logreg', name='Logistic Regression', param_grid={}, gs_cv='split', gs_params={}, **kwargs):
        '''
        Build a Logistic Regression model

        Parameters
        ----------
        key : str
            Unique model identifier
        name : str
            Model name
        param_grid : dict
            Parameter grid for GridSearchCV to go over
        gs_cv : str
            Cross validation option. Must be "split", int, cross-validation generator, an iterable or None. Split will use the object's training and validation indices. For other values, please check sklearn.model_selection.GridSearchCV's cv
        gs_params : dict
            GridSearchCV parameters
        '''
        from sklearn.linear_model import LogisticRegression

        skmodel = LogisticRegression(**kwargs)
        if param_grid:
            self.gridsearchcv(skmodel, key, name, param_grid, cv=gs_cv, **gs_params)
        else:
            self.train_sklearn_model(skmodel, key, name=name)

    def knn(self, key='knn', name='K-Nearest Neighbors', param_grid={}, gs_cv='split', gs_params={}, **kwargs):
        '''
        Build a K-Nearest Neighbors model

        Parameters
        ----------
        key : str
            Unique model identifier
        name : str
            Model name
        param_grid : dict
            Parameter grid for GridSearchCV to go over
        gs_cv : str
            Cross validation option. Must be "split", int, cross-validation generator, an iterable or None. Split will use the object's training and validation indices. For other values, please check sklearn.model_selection.GridSearchCV's cv
        gs_params : dict
            GridSearchCV parameters
        '''
        from sklearn.neighbors import KNeighborsClassifier

        skmodel = KNeighborsClassifier(**kwargs)
        if param_grid:
            self.gridsearchcv(skmodel, key, name, param_grid, cv='split', **gs_params)
        else:
            self.train_sklearn_model(skmodel, key, name=name)

    def nn(self, key='nn', name='Neural Network', param_grid={}, gs_cv='split', gs_params={}, **kwargs):
        '''
        Build a Neural Network (Multi-layer Perceptron)

        Parameters
        ----------
        key : str
            Unique model identifier
        name : str
            Model name
        param_grid : dict
            Parameter grid for GridSearchCV to go over
        gs_cv : str
            Cross validation option. Must be "split", int, cross-validation generator, an iterable or None. Split will use the object's training and validation indices. For other values, please check sklearn.model_selection.GridSearchCV's cv
        gs_params : dict
            GridSearchCV parameters
        '''
        from sklearn.neural_network import MLPClassifier

        skmodel = MLPClassifier(**kwargs)
        if param_grid:
            self.gridsearchcv(skmodel, key, name, param_grid, cv=gs_cv, **gs_params)
        else:
            self.train_sklearn_model(skmodel, key, name=name)
            
    def get_coord_and_label(self, value):
        '''
        Get coordinate values and label
        
        Parameters
        ----------
        value : str or int
            Feature name or coordinate index
            
        Returns
        -------
        array
            Coordinate values
        str
            Coordinate label
        '''
        if isinstance(value, str):
            label = value
            values = self.get_M_feature(value)
        else:
            label = self.coordinate_labels[value]
            values = self.get_coordinates_by_index(value)
        return values, label
        
    def scatter_validation(self, model='', x=0, y=0, bgcolor='lightgrey', bgsize=1, bgalpha=1, alpha=1, \
                       dotsize=5, linewidth=0, removeticks=True, colormap='tab10', \
                       lgdloc='upper center', lgdncol=4, lgdbbox_to_anchor=(0.5, 0), \
                       lgdframeon=True, savefig='', dpi=200, onelimit=False, cmap='YlOrRd', \
                       gridsize=18):
        '''
        Plot scatter plots of training and validation data, with true and predicted labels
        
        Parameters
        ----------
        model : str
        x : int
        y : int
        bgcolor : str
        bgsize : int or float
        bgalpha : int or float
        alpha : int or float
        dotsize : int or float
        linewidth : int or float
        removeticks : bool
        colormap : str
        lgdloc : str
        lgdncol : int
        lgdbbox_to_anchor : tuple
        lgdframeon : bool
        savefig : str
        dpi : int
        onelimit : bool
        cmap : str
        '''
        from matplotlib.patches import Patch
        import seaborn as sns

        x, xlabel = self.get_coord_and_label(x)
        y, ylabel = self.get_coord_and_label(y)
        
        acc = np.round(self.models[model].accuracy*100,2)
        X_val, lbls_val = self.get_validation_data()
        predictions = self.models[model].predict(X_val)
        prediction_probabilities = self.models[model].predict_proba(X_val)
        prediction_probabilities = prediction_probabilities.max(axis=1)
        train_idx, val_idx = self.get_train_val_indices()
        labels = self.target
        
        # randomization (training indices)
        rdm_idx = np.arange(len(train_idx))
        train_idx = train_idx[rdm_idx]
        
        # randomization (validation indices, true validation labels, predicted labels and predictions probabilities from validation data)
        rdm_idx = np.arange(len(val_idx))
        np.random.shuffle(rdm_idx)
        val_idx = val_idx[rdm_idx]
        lbls_val = lbls_val[rdm_idx]
        predictions = predictions[rdm_idx]
        prediction_probabilities = prediction_probabilities[rdm_idx]

        unique_labels = np.unique(labels)
        palette_colors = sns.color_palette(colormap)
        palette_dict = {ul: color for ul, color in zip(unique_labels, palette_colors)}

        fig, axes = plt.subplots(2, 2)
        ax1, ax2, ax3, ax4 = tuple(axes.flatten())
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_box_aspect(1)

        ax1.scatter(x, y, c=bgcolor, s=bgsize, alpha=bgalpha)
        sns.scatterplot(x=x[train_idx], y=y[train_idx], hue=labels[train_idx], ax=ax1, s=dotsize, linewidth=linewidth, palette=palette_dict, alpha=alpha)    
        ax1.get_legend().remove()
        ax1.set_title('Training data')
        ax1.set_ylabel(ylabel)

        ax2.scatter(x, y, c=bgcolor, s=bgsize, alpha=bgalpha)
        sns.scatterplot(x=x[val_idx], y=y[val_idx], hue=lbls_val, ax=ax2, s=dotsize, linewidth=linewidth, palette=palette_dict, alpha=alpha)
        ax2.get_legend().remove()
        ax2.set_title('Validation data\ntrue labels')

        ax3.scatter(x, y, c=bgcolor, s=bgsize, alpha=bgalpha)
        sns.scatterplot(x=x[val_idx], y=y[val_idx], hue=predictions, ax=ax3, s=dotsize, linewidth=linewidth, palette=palette_dict, alpha=alpha)
        ax3.get_legend().remove()
        ax3.set_title(f'Validation data\npredicted labels ({acc}% accuracy)')
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        
        if onelimit:
            norm = plt.Normalize(0, 1)
        else:
            norm = plt.Normalize(0, prediction_probabilities.max())
            
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        ax4.hexbin(x=x[val_idx], y=y[val_idx],C=prediction_probabilities,gridsize=gridsize,cmap=cmap)
        ax4.set_facecolor('black')
        plt.colorbar(mappable=sm, ax=ax4, pad=0.083)
        ax4.set_title('Prediction probabilities')
        ax4.set_xlabel(xlabel)

        handles, labels = ax1.get_legend_handles_labels()
        handles = [
            Patch(facecolor=handle.get_facecolor(), label=label)
            for handle, label in zip(handles, labels)
        ]
        fig.legend(handles, labels, loc=lgdloc, ncol=lgdncol, bbox_to_anchor=lgdbbox_to_anchor, frameon=lgdframeon, handlelength=1.5, handleheight=1)

        if removeticks:
            for ax in [ax1, ax2, ax3, ax4]:  
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=dpi)
        plt.close()
            
    def scatter_feature(self, feature, x=0, y=1, cmap='magma_r', savefig='', dpi=200):
        x, xlabel = self.get_coord_and_label(x)
        y, ylabel = self.get_coord_and_label(y)
        c = self.get_M_feature(feature)
        i = np.argsort(c)
        plt.scatter(x[i], y[i], c=c[i], cmap=cmap)
        plt.colorbar()

        plt.gca().set_box_aspect(1)
        plt.xticks([],[])
        plt.yticks([],[])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(feature)

        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=dpi)
        plt.close()

    def delete(self, key=''):
        '''
        Delete a given model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        '''
        try:
            print(f'Deleting model: {key}')
            del self.models[key]
            print(f'Model {key} deleted succesfully\n')
        except:
            raise Exception(f'Invalid key: {key}\n')
        
    def save(self, file):
        '''
        Save Predictor object
        
        Parameters
        ----------
        file : str
            Path to output file
        '''
        import pickle

        with open(file,'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
    def save_model(self, key, file):
        '''
        Save given model
        
        Parameters
        ----------
        key : str
            Unique model identifier
        file : str
            Path to ouptut file
        '''
        self.models[key].save(file)
    
    def __str__(self):
        s = 'Predictor object\n'
        s += '----------------\n'
        s += f'Data: {self.shape[0]} samples, {self.shape[1]} features\n'
        s += f'Training: {len(self.train_idx)} samples ({self.pct_train*100}%) \n'
        s += f'Validation: {len(self.val_idx)} samples\n'
        s += '----------------\n'
        s += f'Available models: {len(self.models)}\n'
        if len(self.models) != 0:
            for k in self.models.keys():
                s += f'\t- {str(self.models[k])}\n'
        return s