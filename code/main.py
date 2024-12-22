import gc
import random
import csv
from model_evaluation_metrics import cv_model_evaluate
from adjacency_matrix_construction_and_utils import *
from model import HGBHANModel
from optimizer import Optimizer
tf.compat.v1.disable_eager_execution()
import os
from sklearn.metrics import roc_curve
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_auc_curve(y_true, y_pred,auc_curve):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc =auc_curve
    with open('fpr_tpr_values.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['FPR', 'TPR'])
        for f, t in zip(fpr, tpr):
            writer.writerow([f, t])
    print(fpr)
    print(tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' %auc_curve )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('auc_curve1.png')
    plt.show()


def PredictScore(train_drug_microbe_matrix, drug_matrix, microbe_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    adj = constructHNet(train_drug_microbe_matrix, drug_matrix, microbe_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_microbe_matrix.sum()
    X = constructNet(train_drug_microbe_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_microbe_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))
    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]

    placeholders = \
        {
            'features': tf.compat.v1.sparse_placeholder(tf.compat.v1.float32),
            'adj': tf.compat.v1.sparse_placeholder(tf.compat.v1.float32),
            'adj_orig': tf.compat.v1.sparse_placeholder(tf.compat.v1.float32),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
            'adjacency_dropout_rate': tf.compat.v1.placeholder_with_default(0., shape=())
        }

    model = HGBHANModel(placeholders, num_features, emb_dim,
                        features_nonzero, adj_nonzero, train_drug_microbe_matrix.shape[0], name='HGBHAN')

    with tf.compat.v1.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.compat.v1.reshape(tf.compat.v1.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_microbe_matrix.shape[0], num_v=train_drug_microbe_matrix.shape[1],
            association_nam=association_nam)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjacency_dropout_rate']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})  # 关闭 Dropout
            feed_dict.update({placeholders['adjacency_dropout_rate']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))

    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjacency_dropout_rate']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res


def cross_validation_experiment_with_auc(drug_microbe_matrix, drug_matrix, microbe_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(drug_microbe_matrix == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 10
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating drug-microbe...." % (seed))
    answer = np.zeros(shape=(1, 160862))

    y_true_all = []
    y_pred_all = []

    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_microbe_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_microbe_matrix.shape[0]
        microbe_len = drug_microbe_matrix.shape[1]
        drug_microbe_res = PredictScore(
            train_matrix, drug_matrix, microbe_matrix, seed, epochs, emb_dim, dp, lr, adjdp)

        predict_y_proba = drug_microbe_res.reshape(drug_len, microbe_len)
        metric_tmp = cv_model_evaluate(
            drug_microbe_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        metric += metric_tmp
        y_true = drug_microbe_matrix.flatten()
        y_pred = predict_y_proba.flatten()
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        del train_matrix
        gc.collect()

    print(metric / k_folds)
    metric = np.array(metric / k_folds)
    return metric, np.array(y_true_all), np.array(y_pred_all)



if __name__ == "__main__":
    drug_sim = np.loadtxt('../data/MDAD/drug_similarity.csv', delimiter=',')
    microbe_sim = np.loadtxt('../data/MDAD/microbe_similarity.csv', delimiter=',')
    drug_microbe_matrix = np.loadtxt('../data/MDAD/drug_microbe_similarity.txt')


    num_epochs = 600
    embedding_dimension = 128
    learning_rate = 0.01
    adjacency_dropout_rate = 0.6
    dropout_rate = 0.4
    similarity_weight = 6
    result_array = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    num_iterations = 1
    seed = int.from_bytes(os.urandom(4), 'big')
    for i in range(num_iterations):
        metric, y_true, y_pred = cross_validation_experiment_with_auc(
            drug_microbe_matrix, drug_sim * similarity_weight, microbe_sim * similarity_weight, seed, num_epochs, embedding_dimension, dropout_rate, learning_rate, adjacency_dropout_rate)

    print(metric)

    plot_auc_curve(y_true, y_pred,metric[0,1])




