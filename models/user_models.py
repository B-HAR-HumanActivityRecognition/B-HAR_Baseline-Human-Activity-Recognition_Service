import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.layers import Flatten, Dense
from kerastuner import HyperModel
from keras.optimizers import Adam
import pickle
import importlib.util
import sys

class DLUserHyperModel(HyperModel):
    """
    The method build must be implemented by the user, he has to put the specific of its model according to perform
    hyper tuning
    """
    def __init__(self, bhar_username: str, name: str, file_py: str, input_shape: tuple, num_classes: int, run: str):
        self.bhar_username = bhar_username
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.__type = 'dl-hyper-user'
        self.name = name
        self.file_py = file_py
        self.__best_model = None
        self.__best_params = None
        self.run = run

    def build(self, hp):
        """
        Implement your custom model here
        :param hp:
        :return:
        """
        spec = importlib.util.spec_from_file_location("module.name", f"storage/{self.bhar_username}/{self.file_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = module
        spec.loader.exec_module(module)
        return module.build(hp)

    def get_type(self):
        return self.__type

    def get_best_model(self, return_params: bool = False):
        if self.__best_model is not None:
            if return_params:
                return self.__best_model, self.__best_params
            else:
                return self.__best_model

    def find_best_model(self, x_train: np.ndarray, y_train: np.ndarray):
        # Put data in the correct format
        X_train = x_train.reshape((-1, self.input_shape[0], self.input_shape[1]))
        y_train = y_train

        # Create the tuner
        tuner = kt.Hyperband(
            self.build,
            objective='val_accuracy',
            max_epochs=30,
            factor=3,
            directory=f'tests/{self.bhar_username}_test/tuning',
            project_name='bhar_{}_{}'.format(self.run, self.name)
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(
            X_train,
            y_train,
            #epochs=100,
            validation_split=0.2,
            callbacks=[stop_early],
            verbose=0
        )

        # Get best hyper parameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyper parameters
        best_model = tuner.hypermodel.build(best_hps)

        # Train
        history = best_model.fit(X_train, y_train, 200)

        # Find best epoch
        val_acc = history.history['accuracy']
        best_epoch = val_acc.index(max(val_acc)) + 1

        # Build the best model
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(X_train, y_train, best_epoch)

        self.__best_model = best_model

    # whole model
    def save_model(self):
        self.__best_model.save(f"{self.bhar_username}_{self.name}")

    # weights only
    def export_model(self):
        self.__best_model.save_weights(f"{self.bhar_username}_{self.name}_weights.h5")


class DLUserModel(object):
    def __init__(
            self, bhar_username: str, name: str, model, input_shape: tuple, num_classes: int, epochs: int = None, is_trained: bool = False
    ):
        self.bhar_username = bhar_username
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.__type = 'dl-user'
        self.name = name
        self.model = model
        self.epochs = epochs
        self.is_trained = is_trained

    def get_type(self):
        return self.__type

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int):
        self.model.fit(x_train, y_train, epochs)

    def test(self, x_test: np.ndarray):
        y_pred = self.model.predict(x_test)
        return y_pred
    
    # whole model
    def save_model(self):
        self.model.save(f"{self.bhar_username}_{self.name}_model.h5")
    
    # weights only
    def export_model(self):
        self.model.save_weights(f"{self.bhar_username}_{self.name}_weights.h5")

class MLUserModel(object):
    def __init__(self, bhar_username: str, name: str, model, is_trained: bool = False):
        self.bhar_username = bhar_username
        self.__type = 'ml-user'
        self.name = name
        self.model = model
        self.is_trained = is_trained

    def get_type(self):
        return self.__type

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train)

    def test(self, x_test: np.ndarray):
        y_pred = self.model.predict(x_test)
        return y_pred
    
    def get_model(self):
        return self.model

    def export_model(self):
        with open(f'{self.bhar_username}_{self.name}.pkl', 'wb') as file:
            pickle.dump(self.model, file)