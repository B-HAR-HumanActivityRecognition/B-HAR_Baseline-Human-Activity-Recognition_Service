from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle


class BHARMachineLearningModel(object):
    def __init__(self, bhar_username, model, name, tuning_params):
        self.bhar_username = bhar_username
        self.__model = model
        self.name = name
        self.__best_model = None
        self.__best_params = None
        self.__type = 'ml'
        self.__tuning_params = tuning_params

    def get_type(self):
        return self.__type

    def get_name(self):
        return self.name

    def get_model(self):
        return self.__model

    def get_tuning_params(self):
        return self.__tuning_params

    def get_best_model(self, return_params: bool = False):
        if self.__best_model is not None:
            if return_params:
                return self.__best_model, self.__best_params
            else:
                return self.__best_model

    def set_best_model(self, best_model):
        self.__best_model = best_model

    def set_best_params(self, best_params):
        self.__best_params = best_params

    def find_best_model(self, x_train: np.ndarray, y_train: np.ndarray):
        # Define grid search
        clf = GridSearchCV(self.get_model(), self.get_tuning_params(), scoring='accuracy', n_jobs=-1)

        # Search best model
        clf.fit(x_train, y_train)

        # Save best model and params
        self.set_best_model(clf.best_estimator_)
        self.set_best_params(clf.best_params_)

    def export_model(self):
        with open(f'{self.bhar_username}_{self.name}.pkl', 'wb') as file:  
            pickle.dump(self.__best_model, file)


class KNN(BHARMachineLearningModel):
    def __init__(self, bhar_username):
        super().__init__(
            bhar_username=bhar_username,
            model=KNeighborsClassifier(n_jobs=-1),
            name='K-Nearest Neighbors',
            tuning_params={
                'n_neighbors': [1, 3, 5],
                'p': [1, 3, 5],
                'metric': ['euclidean', 'manhattan']
            }
        )


class RandomForest(BHARMachineLearningModel):
    def __init__(self, bhar_username):
        super().__init__(
            bhar_username=bhar_username,
            model=RandomForestClassifier(n_jobs=-1),
            name='Random Forest',
            tuning_params={
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2'],
                'min_samples_split': [2, 3, 5]
            }
        )


class SupportVectorClassification(BHARMachineLearningModel):
    def __init__(self, bhar_username):
        super().__init__(
            bhar_username=bhar_username,
            model=SVC(),
            name='SVC',
            tuning_params={
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        )


class WKNN(BHARMachineLearningModel):
    def __init__(self, bhar_username):
        super().__init__(
            bhar_username=bhar_username,
            model=KNeighborsClassifier(weights='distance', n_jobs=-1),
            name='WKNN',
            tuning_params={
                'n_neighbors': list(range(1, 21)),
                'p': [1, 2, 3, 4, 5],
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
            }
        )


class LDA(BHARMachineLearningModel):
    def __init__(self, bhar_username):
        super().__init__(
            bhar_username=bhar_username,
            model=LinearDiscriminantAnalysis(),
            name='LDA',
            tuning_params={
                'solver': ['svd'],
                'store_covariance': ['True', 'False'],
                'tol': [0.0001, 0.001, 0.01]
            }
        )


class QDA(BHARMachineLearningModel):
    def __init__(self, bhar_username):
        super().__init__(
            bhar_username=bhar_username,
            model=QuadraticDiscriminantAnalysis(),
            name='QDA',
            tuning_params={
                'store_covariance': ['True', 'False'],
                'tol': [0.0001, 0.001, 0.01]
            }
        )


class DecisionTree(BHARMachineLearningModel):
    def __init__(self, bhar_username):
        super().__init__(
            bhar_username=bhar_username,
            model=DecisionTreeClassifier(),
            name='Decision Tree',
            tuning_params={
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'min_samples_split': [2, 3, 5],
                'max_features': ['sqrt', 'log2']
            }
        )