from graph_convolution_models import GraphAttentionConvolution, GraphConvolutionSparse, InnerProductDecoder, GraphAttentionBiLSTMConvolution
from adjacency_matrix_construction_and_utils import *
from keras.layers import LayerNormalization

class HGBHANModel():
    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, name,
                 act=tf.compat.v1.nn.elu, use_hidden1=True, use_hidden2=True, use_hidden3=False):

        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjacency_dropout_rate']
        self.act = act
        self.att = tf.compat.v1.Variable(tf.compat.v1.constant([0.9, 0.45, 0.4]))
        self.num_r = num_r
        self.layer_norm = LayerNormalization(axis=1)
        self.use_hidden1 = use_hidden1
        self.use_hidden2 = use_hidden2
        self.use_hidden3 = use_hidden3

        with tf.compat.v1.variable_scope(self.name):
            self.build()

    def build(self):
        self.adj = dropout_sparse(self.adj, 1 - self.adjdp, self.adj_nonzero)

        if self.use_hidden1:
            self.hidden1 = GraphConvolutionSparse(
                name='gcn_sparse_layer',
                input_dim=self.input_dim,
                output_dim=self.emb_dim,
                adj=self.adj,
                features_nonzero=self.features_nonzero,
                dropout=self.dropout,
                act=self.act
            )(self.inputs)
        else:
            self.hidden1 = self.inputs

        if self.use_hidden2:
            self.hidden2 = GraphAttentionBiLSTMConvolution(
                name='gat_dense_layer',
                input_dim=self.emb_dim,
                output_dim=self.emb_dim,
                dropout=self.dropout
            )(self.hidden1, self.adj)
        else:
            self.hidden2 = self.hidden1

        if self.use_hidden3:
            self.hidden3 = GraphAttentionConvolution(
                name='gat_dense_layer1',
                former=self.hidden1,
                input_dim=self.emb_dim,
                output_dim=self.emb_dim,
                dropout=self.dropout
            )(self.hidden2, self.adj)
        else:
            self.hidden3 = self.hidden2

        self.embeddings = self.hidden1 * self.att[0] + self.hidden2 * self.att[1] + self.hidden3 * self.att[2]

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim,
            num_r=self.num_r,
            act=tf.compat.v1.nn.sigmoid
        )(self.embeddings)
