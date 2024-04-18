import numpy as np
from copy import deepcopy

class ActiveLearner:
    def __init__(self, X, y, model, n_samples, n_queries, strategy, human_error = 0.0):
        # Training data and labels
        self.X = X
        self.y = y
        
        # Active learning model
        self.model = model
        self.model_init = deepcopy(self.model)
        
        # Number of samples to be queried at each iteration
        self.n_samples = n_samples
        
        # Number of Active Learning iterations
        self.n_queries = n_queries
        
        # Active learning strategy
        self.strategy = strategy
        assert self.strategy in ['random', 'least_conf', 'margin_conf', 'ratio_conf', 'entropy'], 'Invalid strategy'
        
        # Human error
        self.human_error = human_error
        
        # Data pools for active learning
        self.X_pool = deepcopy(self.X)
        self.y_pool = deepcopy(self.y)
    
    def reset(self):
        # Reset model
        self.model = deepcopy(self.model_init)
        
        # Reset data pools
        self.X_pool = deepcopy(self.X)
        self.y_pool = deepcopy(self.y)
    
    def _query(self, n_samples, init = False):
        def _get_labels(idx):
            # We simulate human labeling by correctly labeling the samples based on true labels
            y_query = self.y_pool[idx]
            
            # Simulate human labeling error by randomly changing labels
            if self.human_error > 0:
                idx_mask = np.random.rand(len(y_query)) < self.human_error
                y_query[idx_mask] = np.random.choice(np.unique(self.y), size = np.sum(idx_mask), replace = True)
            
            return y_query
        
        if init:
            # Randomly select samples for initial labeling
            idx = np.random.choice(range(len(self.X_pool)), size = n_samples, replace = False)
            
            # Make sure, that all classes are present
            while len(np.unique(self.y_pool[idx])) < len(np.unique(self.y)):
                idx = np.random.choice(range(len(self.X_pool)), size = n_samples, replace = False)
        else:
            # Predict probabilities for each class
            y_prob = self.model.predict_proba(self.X_pool)
        
            if self.strategy == 'random' or init:
                # Randomly select samples
                idx = np.random.choice(range(len(self.X_pool)), size = n_samples, replace = False)
                
                # If init, make sure, that all classes are present
                if init:
                    y_unique = len(np.unique(self.y_pool[idx]))
                    while y_unique < len(np.unique(self.y)):
                        idx = np.random.choice(range(len(self.X_pool)), size = n_samples, replace = False)
                        y_unique = len(np.unique(self.y_pool[idx]))     
            elif self.strategy == 'least_conf':
                # Difference between the most confident prediction and 100% confidence
                most_conf = np.nanmax(y_prob, axis = 1)
                numerator = (y_prob.size * (1 - most_conf))
                denominator = (y_prob.size - 1)
                idx = np.argpartition((numerator / denominator), -n_samples)[-n_samples:]
            elif self.strategy == 'margin_conf':
                # Difference between the top two most confident predictions
                raise NotImplementedError('Margin Confidence sampling strategy is not implemented yet ,-)')
            elif self.strategy == 'ratio_conf':
                # Ratio between the top two most confident predictions
                raise NotImplementedError('Ratio Confidence sampling strategy is not implemented yet ,-)')
            elif self.strategy == 'entropy':
                # Difference between all predictions based on entropy, as defined by information theory
                raise NotImplementedError('Entropy sampling strategy is not implemented yet ,-)')
        
        # Retrieve queried samples from data pool
        X_query = self.X_pool[idx]
        
        # Get human labeled data based on queried samples
        y_query = _get_labels(idx)
        
        # Remove queried samples from data pool
        self.X_pool = np.delete(self.X_pool, idx, axis = 0)
        self.y_pool = np.delete(self.y_pool, idx, axis = 0)
            
        return X_query, y_query
    
    def fit(self):
        # Reset model and data pools
        self.reset()
        
        # History of models
        history = {}
        
        # Fit model on randomly queried training data (simulates pre-labeled data)
        X_train, y_train = self._query(self.n_samples, init = True)
        self.model.fit(X_train, y_train)
        
        history.update({
            0 : {
                'model': deepcopy(self.model),
                'X': X_train,
                'y': y_train
            }
        })
        
        # Active learning iterations
        for i in range(self.n_queries):
            # Check if enough samples are left for labeling
            if len(self.X_pool) == 0:
                break
            elif len(self.X_pool) < self.n_samples:
                n_samples = len(self.X_pool)
            else:
                n_samples = self.n_samples
            
            # Query samples based on selected strategy
            X_query, y_query = self._query(n_samples)
            
            # Add queried samples to training data
            X_train = np.concatenate((X_train, X_query))
            y_train = np.concatenate((y_train, y_query))
            
            # Fit model on queried samples
            self.model.fit(X_train, y_train)
            history.update({
                i+1 : {
                    'model': deepcopy(self.model),
                    'X': X_train,
                    'y': y_train
                }
            })
        
        # Return model
        return history
