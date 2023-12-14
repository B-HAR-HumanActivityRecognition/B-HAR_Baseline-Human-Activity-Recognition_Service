import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.optimizers import Adam


def build(hp):
    model = tf.keras.Sequential()
    model.add(
        Dense(
            units=hp.Int(
                'units',
                min_value=50,
                max_value=100,
                step=25
            ),
            activation=hp.Choice(
                'dense_activation',
                values=['relu', 'tanh', 'sigmoid'],
                default='relu'
            )
        )
    )
    model.add(
        Dense(
            units=hp.Int(
                'units',
                min_value=50,
                max_value=100,
                step=25
            ),
            activation=hp.Choice(
                'dense_activation',
                values=['relu', 'tanh', 'sigmoid'],
                default='relu'
            )
        )
    )
    model.add(
        Dense(
            units=hp.Int(
                'units',
                min_value=50,
                max_value=100,
                step=25
            ),
            activation=hp.Choice(
                'dense_activation',
                values=['relu', 'tanh', 'sigmoid'],
                default='relu'
            )
        )
    )
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))
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