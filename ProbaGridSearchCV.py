import sklearn.grid_search

class ProbaGridSearchCV(sklearn.grid_search.GridSearchCV):

    @property
    def predict_proba(self):
        if hasattr(self, 'best_estimator_'):
            return super(ProbaGridSearchCV, self).predict_proba
        return None

    @property
    def decision_function(self):
        if hasattr(self, 'best_estimator_'):
            return super(ProbaGridSearchCV, self).decision_function
        return None
