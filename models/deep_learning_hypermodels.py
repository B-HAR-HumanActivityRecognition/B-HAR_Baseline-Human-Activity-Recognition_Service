import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, SimpleRNN, LSTM
from kerastuner import HyperModel
from keras.optimizers import Adam


class BHARDeepLearningModel(HyperModel):
    def __init__(self, bhar_username: str, name: str, model_type: str,  input_shape: tuple, num_classes: int, run: str):
        self.bhar_username = bhar_username
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.__type = model_type
        self.name = name
        self.run_name = run
        self.__best_model = None
        self.__best_params = None

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
            project_name='bhar_{}_{}'.format(self.run_name, self.name)
        )

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(
            X_train,
            y_train,
            #epochs=30,
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

    def build(self, hp):
        pass

    # whole model
    def save_model(self):
        self.__best_model.save(f"{self.bhar_username}_{self.name}_model.h5")

    # weights only
    def export_model(self):
        self.__best_model.save_weights(f"{self.bhar_username}_{self.name}_weights.h5")

class CNNHyperModel(BHARDeepLearningModel):
    """
    The structure of this model is the following:
    Conv1D -> Conv1D -> MaxPool1D -> Conv1D -> Conv1D -> MaxPool1D -> Flatten -> Dense -> Dense

    """
    def __init__(self, bhar_username:str, input_shape: tuple, num_classes: int, run: str):
        super().__init__(
            bhar_username=bhar_username,
            name='CNN Big',
            input_shape=input_shape,
            num_classes=num_classes,
            model_type='dl',
            run=run
        )

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(
            Conv1D(
                filters=16,
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(
            Conv1D(
                filters=16,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv1D(
                filters=32,
                kernel_size=3,
                activation='relu'
            )
        )
        model.add(
            Conv1D(
                filters=hp.Choice(
                    'num_filters',
                    values=[32, 64],
                    default=64,
                ),
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(Flatten())
        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(
                    'dropout_3',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


class LightCNNHyperModel(BHARDeepLearningModel):
    """
    The structure of this model is the following:
    Conv1D -> MaxPool1D -> Conv1D -> MaxPool1D -> Flatten -> Dense -> Dense

    """
    def __init__(self, bhar_username: str, input_shape: tuple, num_classes: int, run: str):
        super().__init__(
            bhar_username=bhar_username,
            name='CNN Light',
            input_shape=input_shape,
            num_classes=num_classes,
            model_type='dl',
            run=run
        )

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(
            Conv1D(
                filters=16,
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv1D(
                filters=32,
                kernel_size=3,
                activation='relu'
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(Flatten())
        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(
                    'dropout_3',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
class RNNHyperModel(BHARDeepLearningModel):
    """
    The structure of this model is the following:
    SimpleRNN -> SimpleRNN -> SimpleRNN -> Dense
    
    """
    def __init__(self, bhar_username: str, input_shape: tuple, num_classes: int, run: str):
        super().__init__(
            bhar_username=bhar_username,
            name='RNN',
            input_shape=input_shape,
            num_classes=num_classes,
            model_type='dl',
            run=run
        )

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(
            SimpleRNN(
                units=hp.Int(
                    'input_unit',
                    min_value=20,
                    max_value=60,
                    step=20
                ),
                return_sequences=True,
                dropout=hp.Float('in_dropout', min_value=0.0, max_value=0.5, step=0.1),
                input_shape=self.input_shape
            )
        )
        model.add(
            SimpleRNN(
                units=hp.Int(
                    'layer_1',
                    min_value=20,
                    max_value=60,
                    step=20
                ),
                return_sequences=True,
                activation=hp.Choice(
                    'l1_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                ),
                dropout=hp.Float('l1_dropout', min_value=0.0, max_value=0.5, step=0.1),
            )
        )
        model.add(
            SimpleRNN(
                units=hp.Int(
                    'layer_2',
                    min_value=20,
                    max_value=60,
                    step=20
                ),
                activation=hp.Choice(
                    'l2_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                ),
                dropout=hp.Float('l2_dropout', min_value=0.0, max_value=0.5, step=0.1),
            )
        )
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
class LSTMHyperModel(BHARDeepLearningModel):
    """
    The structure of this model is the following:
    LSTM -> LSTM -> LSTM -> Dense
    
    """
    def __init__(self, bhar_username: str,input_shape: tuple, num_classes: int, run: str):
        super().__init__(
            bhar_username=bhar_username,
            name='LSTM',
            input_shape=input_shape,
            num_classes=num_classes,
            model_type='dl',
            run=run
        )

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(
            LSTM(
                units=hp.Int(
                    'input_unit',
                    min_value=20,
                    max_value=60,
                    step=20
                ),
                return_sequences=True,
                dropout=hp.Float('in_dropout', min_value=0.0, max_value=0.5, step=0.1),
                input_shape=self.input_shape
            )
        )
        model.add(
            LSTM(
                units=hp.Int(
                    'layer_1',
                    min_value=20,
                    max_value=60,
                    step=20
                ),
                return_sequences=True,
                activation=hp.Choice(
                    'l1_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                ),
                dropout=hp.Float('l1_dropout', min_value=0.0, max_value=0.5, step=0.1),
            )
        )
        model.add(
            LSTM(
                units=hp.Int(
                    'layer_2',
                    min_value=20,
                    max_value=60,
                    step=20
                ),
                activation=hp.Choice(
                    'l2_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                ),
                dropout=hp.Float('l2_dropout', min_value=0.0, max_value=0.5, step=0.1),
            )
        )
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model