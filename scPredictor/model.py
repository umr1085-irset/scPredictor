class Model:
    '''
    A class used to represent an ML model
    
    Attributes
    ----------
    name : str
        The model name
    key : str
        Unique model identifier
    model : model
        The model (Scikit-Learn, Keras...)
    accuracy : float
        The model accuracy
    class_names : array
        Array of class names used in the model
    report : report
        The model classification report
    report_dict : dict
        The model classification report as a dictionary
        
    Methods
    -------
    plot_proba(arr, cmap='magma', interpolation='none', onelimit=False, cluster='none', savefig='', dpi=200, **kwargs)
        Plots a heatmap of prediction probabilities for each element of array
    plot_proba_validation(arr, lbls, cmap='magma', interpolation='none', onelimit=False, cluster='none', savefig='', dpi=200, **kwargs)
        Plots a heatmap of prediction probabilities with elements split by target
    save(file, method='')
        Save model to file
    '''
    def __init__(self, key, name, model, accuracy, class_names, report, report_dict, cv_results=None, cv_best_params=None):
        '''
        Parameters
        ----------
        name : str
            The model name
        key : str
            Unique model identifier
        model : model
            The model (Scikit-Learn, Keras...)
        accuracy : float
            The model accuracy
        class_names : array
            Array of class names used in the model
        report : report
            The model classification report
        report_dict : dict
            The model classification report as a dictionary   
        cv_results : dict
            Cross Validation results if grid search was run
        '''
        self.name = name
        self.key = key
        self.model = model
        self.accuracy = accuracy
        self.class_names = class_names
        self.report = report
        self.report_dict = report_dict
        self.cv_results = cv_results
        self.cv_best_params = cv_best_params
    
    def predict(self, arr):
        try:
            return self.model.predict(arr)
        except:
            return self.model.predict(arr.toarray())
    
    def predict_proba(self, arr):
        try:
            return self.model.predict_proba(arr)
        except:
            return self.model.predict_proba(arr.toarray())
        
    def save(self, file):
        '''
        Save model to file
        
        Parameters
        ----------
        file : str
            Path to output file
        '''
        import pickle 
        
        with open(file,'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)
        
    def __str__(self):
        s = f'["{self.key}"] {self.name}: {self.accuracy*100:.2f}% accuracy'
        return s
