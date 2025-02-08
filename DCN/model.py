"""
Created on Feb 8, 2025

model: Deep & Cross Network

Author: Ruikai Liang
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Layer, Dropout


class CrossLayer(Layer):
    """
    Cross Network
    """

    def __init__(self, layer_num, reg_w=1e-4, reg_b=1e-4):
        """
        Init method
        :param layer_num: int, the number of cross layers
        :param reg_w: float, the regularization coefficient of weights
        :param reg_b: float, the regularization coefficient of bias
        """
        super(CrossLayer, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_uniform',
                            regularizer=l2(self.reg_w),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer='random_uniform',
                            regularizer=l2(self.reg_b),
                            trainable=True
                            )
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

class DNN(Layer):
    """
    Deep Neural Network
    """
    
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """
        Init method
        :param hidden_units: list, the number of hidden units in each layer
        :param activation: str, the activation function
        :param dropout: float, the dropout rate
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class DCN(keras.Model):
    """
    Deep & Cross Network
    """

    def __init__(self, feature_columns, hidden_units, activation='relu', dropout=0., embed_reg=1e-4, cross_w_reg=1e-4, cross_b_reg=1e-4):
        super(DCN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.layer_num = len(hidden_units)
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossLayer(self.layer_num, cross_w_reg, cross_b_reg)
        self.dnn_network = DNN(hidden_units, activation, dropout)
        self.dense_final = Dense(1)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)
        # Cross Network
        cross_out = self.cross_network(x)
        # DNN
        dnn_out = self.dnn_network(x)
        total_out = self.dense_final(tf.concat([cross_out, dnn_out], axis=-1))
        return total_out

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()
        

    
