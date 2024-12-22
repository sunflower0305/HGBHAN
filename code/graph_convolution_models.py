from adjacency_matrix_construction_and_utils import *
import tensorflow as tf
from keras.layers import LayerNormalization
from keras import layers


class GraphAttentionBiLSTMConvolution():
    def __init__(self, input_dim, output_dim, name, dropout=0.2, concat=True, act=tf.nn.relu):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vars = {}
        self.issparse = False
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.alpha = 0.2
        self.concat = concat
        self.bi_lstm = layers.Bidirectional(layers.LSTM(units=int(self.output_dim / 2), input_shape=(10, self.input_dim)))
        self.W = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)))
        self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(2 * output_dim, 1)))
        self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
        self.layer = LayerNormalization(axis=1)
        self.act = act

    def __call__(self, h, adj, training=True):
        with tf.compat.v1.name_scope(self.name):
            h = tf.expand_dims(h, axis=-1)
            h = self.bi_lstm(h)
            Wh = tf.matmul(h, self.W)
            e = self._prepare_attentional_mechanism_input(Wh)
            zero_vec = -9e15 * tf.ones_like(e)
            adj = tf.sparse.to_dense(tf.sparse.reorder(adj))
            adj = adj + tf.eye(tf.shape(adj)[0])
            attention = tf.where(adj > 0, e, zero_vec)
            attention = tf.nn.softmax(attention, axis=-1)
            attention = tf.nn.dropout(attention, rate=self.dropout)
            h_prime = tf.matmul(attention, Wh)
        return self.leakyrelu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        a_input = tf.concat([Wh, Wh], axis=-1)
        e = tf.matmul(self.leakyrelu(a_input), self.a)
        output = tf.squeeze(e, axis=-1)
        return output


class GraphAttentionConvolution():

    def __init__(self, former, input_dim, output_dim, name, dropout=0.2, concat=True, act=tf.nn.relu):
        self.name = name
        self.input_dim = input_dim
        self.former = former
        self.output_dim = output_dim
        self.vars = {}
        self.issparse = False
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.alpha = 0.2
        self.concat = concat
        self.W = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(input_dim, output_dim)))
        self.a = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=(2 * output_dim, 1)))
        self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
        self.act = act

    def __call__(self, h, adj, training=True):
        with tf.compat.v1.name_scope(self.name):
            Wh = tf.matmul(h, self.W)
            e = self._prepare_attentional_mechanism_input(Wh)
            zero_vec = -9e15 * tf.ones_like(e)
            adj = tf.sparse.to_dense(tf.sparse.reorder(adj))
            adj = adj + tf.eye(tf.shape(adj)[0])
            attention = tf.where(adj > 0, e, zero_vec)
            attention = tf.nn.softmax(attention, axis=-1)
            attention = tf.nn.dropout(attention, rate=self.dropout)
            h_prime = tf.matmul(attention, Wh)
            h_prime = h_prime + h

        return self.leakyrelu(h_prime) + self.former

    def _prepare_attentional_mechanism_input(self, Wh):
        a_input = tf.concat([Wh, Wh], axis=-1)
        e = tf.matmul(self.leakyrelu(a_input), self.a)
        output = tf.squeeze(e, axis=-1)
        return output


class GraphConvolutionSparse():
    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            x = tf.sparse.sparse_dense_matmul(x, self.vars['weights'])
            x = tf.sparse.sparse_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class InnerProductDecoder():
    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 0.7 - self.dropout)

            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs


