import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from lib.data import Data
from lib.display import Display, TensorBoard
from lib.build import BuildNetwork
import argparse


class Config(object):
    def __init__(self, network, n_train_samples, n_validation_samples, n_epochs, batch_size, embedding_matrix,
                 max_len, embedding_dim, train_x1, train_x2, logdir, contrastive, save_embedding, save_train_data,
                 calculate_validation):
        self.n_train_samples = n_train_samples
        self.n_validation_samples = n_validation_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.embedding_matrix = embedding_matrix
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.logdir = logdir
        self.contrastive = contrastive
        self.calculate_validation = calculate_validation
        self.save_embedding = save_embedding
        self.save_train_data = save_train_data
        self.model = lambda x : BuildNetwork(x, max_len, embedding_dim)
        self.build_network = {
                        "biRNN_fc": lambda y: self.model.build_biRNN_fc_network(y),
                    }[network]

class Run(object):

    def run_network(self, train_csv, config):
        data = Data()
        display = Display()
        data.run(train_csv,
                 n_train_samples=config.n_train_samples,
                 n_validation_samples=config.n_validation_samples,
                 embedding_matrix = config.embedding_matrix,
                 max_len = config.max_len,
                 embedding_dim=config.embedding_dim,
                 train_x1 = config.train_x1,
                 train_x2 = config.train_x2,
                 contrastive=config.contrastive,
                 save_embedding=config.save_embedding,
                 save_train_data=config.save_train_data)
        with tf.Graph().as_default() as graph:
           config.model = config.model(data)
           writer = TensorBoard(graph=graph, logdir=config.logdir).writer
           output, loss, acc, train_summ, valid_summ, opt, merged = config.build_network(graph)
           init = tf.global_variables_initializer()
           with tf.Session(graph=graph) as sess:
               sess.run(init)
               for epoch in range(config.n_epochs):
                 train_iter_ = data.batch_generator(config.batch_size)
                 for batch_idx, batch in enumerate(tqdm(train_iter_)):
                    train_x1_batch, train_x2_batch, train_labels_batch = batch
                    _, batch_train_loss, batch_train_accuracy, batch_train_summary, _, summary = sess.run([output, loss, acc, train_summ, opt, merged],
                                                                                    feed_dict={
                                                                                                config.model.network.x1 : train_x1_batch,
                                                                                                config.model.network.x2 : train_x2_batch,
                                                                                                config.model.loss.labels : train_labels_batch,
                                                                                                config.model.network.embedding_matrix : data.embedding_matrix
                                                                                              })
                    display.log_train(epoch, batch_idx, batch_train_loss, batch_train_accuracy)
                    writer.add_summary(batch_train_summary, batch_idx)

                    if config.calculate_validation:
                        if batch_idx % 100 == 0:
                            batch_valid_accuracy, batch_valid_summary = sess.run([acc, valid_summ], feed_dict={
                                                                                config.model.network.x1 : data.valid_x1,
                                                                                config.model.network.x2 : data.valid_x2,
                                                                                config.model.loss.labels : data.valid_labels,
                                                                                config.model.network.embedding_matrix : data.embedding_matrix
                                                                                })
                            display.log_validation(epoch, batch_idx, batch_valid_accuracy)
                            writer.add_summary(batch_valid_summary, batch_idx)
                    writer.add_summary(summary, batch_idx)

        display.done()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RNN',
                                     formatter_class=argparse.
                                     RawTextHelpFormatter)
    parser.add_argument('-e', '--embedding', required=False,
                        help="Supply embedding matrix file")
    parser.add_argument('-s', '--save_embedding', action="store_true",
                        help="Do you want to save embedding to npy file?")
    parser.add_argument('-d', '--master',  required=False,
                        help="Supply master training dataset")
    parser.add_argument('-x', '--trainx1', required=False,
                        help="Supply training data for q1")
    parser.add_argument('-y', '--trainx2', required=False,
                        help="Supply training data for q1")
    parser.add_argument('-t', '--save_train', action="store_true",
                        help="Do you want to save preprocessed training data to npy file?")
    args = parser.parse_args()

    config = Config(network="biRNN_fc",
                    n_train_samples=500000,
                    n_validation_samples=1000,
                    n_epochs=10,
                    batch_size=100,
                    embedding_matrix = args.embedding,
                    max_len=10,
                    embedding_dim=300,
                    train_x1 = args.trainx1,
                    train_x2 = args.trainx2,
                    logdir="/tmp/quora_logs/biRNN_fc_1",
                    contrastive=False,
                    save_embedding=args.save_embedding,
                    save_train_data=args.save_train,
                    calculate_validation=True)
    Run().run_network(args.master, config)



## TODO:
## * add word2vec/glove functionality
## * add tensorboard optional
## * make it easier to change dimensions
## * cycle validation set
