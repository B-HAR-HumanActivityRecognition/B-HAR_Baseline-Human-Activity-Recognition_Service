# from models.machine_learning_hypermodels import *
# from models.deep_learning_hypermodels import *
# from models.user_models import *
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from utility.reporter import Reporter


class ModelsManager(object):

    def __init__(self, input_shape: int or tuple, num_classes: int, reporter: Reporter, models: dict = None):
        self.__input_shape = input_shape
        self.__num_classes = num_classes
        self.__reporter = reporter
        self.__models = models

    def get_input_shape(self):
        return self.__input_shape

    def get_num_classes(self):
        return self.__num_classes

    def get_models(self):
        """
        Returns the baseline models defined in BHAR
        :return: default models
        """
        return self.__models

    # useless
    # def add_model(self, new_model):
    #     if 'user' not in self.__models.keys():
    #         self.__models['user'] = []

    #     self.__models['user'].append(new_model)

    # useless
    # def drop_model(self, model_name: str, category: str):
    #     for model in self.__models[category]:
    #         if model.get_name() == model_name:
    #             self.__models[category].remove(model)

    def start_train_and_test(
            self,
            x_train: np.ndarray,
            x_test: np.ndarray,
            y_train: np.ndarray,
            y_test: np.ndarray,
            representation: str
    ):
        # Convert target y to numbers
        original, train_coded = np.unique(y_train, return_inverse=True)
        y_train = train_coded
        original, test_coded = np.unique(y_test, return_inverse=True)
        y_test = test_coded
        for category in list(self.__models.keys()):
            self.__reporter.write_section('Evaluating category {}:'.format(category))
            for model in self.__models[category]:
                self.__reporter.write_subsection('Model {}'.format(model.name))
                if model.get_type() == 'ml':
                    # Machine learning format
                    model.find_best_model(x_train, y_train)
                    y_predicted = model.get_best_model().predict(x_test)
                    self.evaluate_model_performances(y_test, y_predicted, original)

                elif (model.get_type() == 'dl' or model.get_type() == 'dl-hyper-user') and representation != 'features':
                    # Deep learning format hyper model
                    shape = (-1, self.get_input_shape()[0], self.get_input_shape()[1])
                    model.find_best_model(x_train, y_train)
                    y_predicted = model.get_best_model().predict(x_test.reshape(shape)).argmax(1)
                    self.evaluate_model_performances(y_test, y_predicted, original)

                elif model.get_type() == 'dl-user':
                    model.train(x_train, y_train, 30)
                    y_predicted = model.test(x_test).argmax(1)
                    self.evaluate_model_performances(y_test, y_predicted, original)

                elif model.get_type() == 'ml-user':
                    model.train(x_train, y_train)
                    y_predicted = model.test(x_test)
                    self.evaluate_model_performances(y_test, y_predicted, original)

    def evaluate_model_performances(self, y_test, y_pred, original) -> None:
        """
        Print the results in terms of sensitivity, precision, recall. etc. achieved by the model

        :param y_test: ground truth values
        :param y_pred: model's predictions
        :param original: target classes (used to print their names in the classification report)
        :return:
        """

        # Metrics
        #f_1_score = f1_score(y_test, y_pred, zero_division=0, average='weighted')
        #precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
        #recall = recall_score(y_test, y_pred, zero_division=0, average='micro')
        #accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=original)
        conf_matrix = np.array2string(confusion_matrix(y_test, y_pred))
        if original.shape[0] == 2:
            roc_fpr, roc_tpr, roc_thrs = roc_curve(y_test, y_pred)
        y_test = y_test.reshape(-1,1)
        y_pred = y_pred.reshape(-1,1)
        encoder = OneHotEncoder()
        y_test_one_hot = encoder.fit_transform(y_test).toarray()
        y_pred_one_hot = encoder.fit_transform(y_pred).toarray()
        roc_auc = roc_auc_score(y_test_one_hot, y_pred_one_hot, multi_class='ovr')


        # tn, fp, _, _ = tuple(confusion_matrix(y_test, y_pred).ravel())
        # specificity = tn / (tn + fp)
        #
        # self.__reporter.write_body('specificity: {}'.format(specificity))
        # self.__reporter.write_body('precision: {}'.format(precision))
        # self.__reporter.write_body('recall: {}'.format(recall))
        # self.__reporter.write_body('f1-score: {}'.format(f_1_score))
        # self.__reporter.write_body('accuracy: {}'.format(accuracy))
        self.__reporter.write_body('classification report: \n{}'.format(class_report))
        self.__reporter.write_body('confusion matrix: \n{}\n'.format(conf_matrix))
        if original.shape[0] == 2:
            self.__reporter.write_body('roc-tpr: {}'.format(roc_tpr))
            self.__reporter.write_body('roc-fpr: {}'.format(roc_fpr))
            self.__reporter.write_body('roc-thrs: {}'.format(roc_thrs))
        else:
            self.__reporter.write_body('roc-auc: {}\n'.format(roc_auc))

    def export_models(self):
        for model_list in self.get_models().values():
            for model in model_list:
                model.export_model()
