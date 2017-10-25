import tensorflow as tf
from lib.architecture import Network
from lib.optimization import *


class BuildNetwork(object):
    def __init__(self, graph, max_len, embedding_dim):
        self.network = Network(graph, max_len=max_len, embedding_dim=embedding_dim)
        self.loss = SoftmaxLoss()
        self.opt = Optimization()
        self.accuracy = Accuracy()

    def build_biRNN_fc_network(self, graph):
        print("building siamese_stacked_fc network...")
        output = self.network.biRNN_fc_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc, train_summ, valid_summ = self.accuracy.sigmoid_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, train_summ, valid_summ, opt, merged
