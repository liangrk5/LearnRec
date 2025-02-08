"""
Created on Feb 8, 2025

train DCN model

Author: Ruikai Liang
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from model import DCN

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    file = '../data/Criteo/train.txt'

    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.2

    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    feature_columns, (train_X, train_y), (test_X, test_y) = create_criteo_dataset(file, embed_dim=embed_dim, read_part=read_part, sample_num=sample_num, test_size=test_size)

    model = DCN(feature_columns, hidden_units, dnn_dropout=dnn_dropout)

    model.summary()

    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=[AUC()])

    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_auc', patience=2, restore_best_weights=True)]
    )

    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])