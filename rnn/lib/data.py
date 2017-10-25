import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing import sequence, text
from gensim.models import KeyedVectors

class Data(object):
    def __init__(self):
        self.train_csv = None
        self.train_x2 = None
        self.train_labels = None
        self.valid_x2 = None
        self.valid_labels = None
        self.test_x2 = None
        self.test_labels = None
        self.train_embedding_matrix = None
        self.valid_embedding_matrix = None
        self.test_embedding_matrix = None

    def import_data(self, train_csv, valid_csv, test_csv):
        print("importing data...")
        train_df = pd.read_csv(train_csv, sep='\t')
        valid_df = pd.read_csv(valid_csv, sep='\t')
        test_df = pd.read_csv(test_csv, sep='\t')
        return train_df, valid_df, test_df

    def preprocess_char(self, df):
        print("preprocessing data...")
        import string
        vocab_chars = string.ascii_lowercase + '0123456789 '
        vocab2ix_dict = {char: (ix+1) for ix, char in enumerate(vocab_chars)}
        vocab_length = len(vocab_chars) + 1
        def sentence2onehot(sentence, vocab2ix_dict = vocab2ix_dict):
            # translate sentence string into indices
            sentence_ix = [vocab2ix_dict[x] for x in list(sentence) if x in vocab_chars]
            # Pad or crop to embedding dimension
            sentence_ix = (sentence_ix + [0]*self.embedding_dim)[0:self.embedding_dim]
            return(sentence_ix)
        self.train_x1 = np.matrix(df.sentence1.str.lower().apply(sentence2onehot).tolist())
        self.train_x2 = np.matrix(df.sentence2.str.lower().apply(sentence2onehot).tolist())
        if self.embedding_matrix is None:
            self.embedding_matrix = tf.diag(tf.ones(shape=[self.embedding_dim]))

        mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        df = df.replace({'gold_label': mapping})

        labels_idx = np.array(df.gold_label)
        self.train_labels = np.zeros((labels_idx.shape[0], 2))
        for i, x in enumerate(labels_idx):
            self.train_labels[i, int(x)] = 1
        return

    def preprocess_word2vec(self, train_df, valid_df, test_df, save_embedding=False, save_train_data=False):
        print("downloading word2vec...")
        word2vec = KeyedVectors.load_word2vec_format('~/Github/quora-kaggle/data/GoogleNews-vectors-negative300.bin',
                                                     binary=True)

        def preprocess(word2vec, df):
            print("preprocessing data...")
            mapping = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
            df = df.replace({'gold_label': mapping})
            if df.gold_label.dtype == 'object':
                df = df.loc[df.gold_label != '-']
            tk = text.Tokenizer(num_words=200000)
            tk.fit_on_texts(list(df.sentence2.values.astype(str)))
            word_index = tk.word_index
            x2 = tk.texts_to_sequences(df.sentence2.values.astype(str))
            x2 = sequence.pad_sequences(x2, maxlen=self.max_len)

            if save_train_data:
                print("saving preprocessed training data...")
                np.save("%s_x1.npy" % self.train_csv, self.train_x1)
                np.save("%s_x2.npy" % self.train_csv, self.train_x2)


            num_words = min(200000, len(word_index)+1)
            print("populating embedding matrix...")
            embedding_matrix = np.zeros((num_words, self.embedding_dim))
            for word, i in word_index.items():
                if word in word2vec.vocab:
                    embedding_matrix[i] = word2vec.word_vec(word)
            return df, x2, embedding_matrix

        def gen_labels(df):
            labels_idx = np.array(df.gold_label)
            labels = np.zeros((labels_idx.shape[0], 3))
            for i, x in enumerate(labels_idx):
                labels[i, int(x)] = 1
            return labels

        train_df, train_x2, train_embedding_matrix = preprocess(word2vec, train_df)
        valid_df, valid_x2, valid_embedding_matrix = preprocess(word2vec, valid_df)
        test_df, test_x2,  test_embedding_matrix = preprocess(word2vec, test_df)

        self.train_x2 = train_x2
        self.valid_x2 = valid_x2
        self.test_x2 = test_x2
        self.train_embedding_matrix = train_embedding_matrix
        self.valid_embedding_matrix = valid_embedding_matrix
        self.test_embedding_matrix = test_embedding_matrix

        self.train_labels = gen_labels(train_df)
        self.valid_labels = gen_labels(valid_df)
        self.test_labels = gen_labels(test_df)

        return

    def subsample(self, n_train_samples, n_validation_samples, n_test_samples):
        print("subsampling data...")
        train_size = self.train_x2.shape[0]
        train_random = np.random.choice(train_size,
                                      n_train_samples,
                                      replace=False)
        np.random.shuffle(train_random)
        train_sample_idx = train_random[:n_train_samples]

        valid_size = self.valid_x2.shape[0]
        valid_random = np.random.choice(valid_size,
                                      n_validation_samples,
                                      replace=False)
        np.random.shuffle(valid_random)
        validation_sample_idx = valid_random[:n_validation_samples]

        test_size = self.test_x2.shape[0]
        test_random = np.random.choice(test_size,
                                       n_test_samples,
                                       replace=False)
        np.random.shuffle(test_random)
        test_sample_idx = test_random[:n_test_samples]

        self.valid_x2 = self.valid_x2[validation_sample_idx, :self.embedding_dim]
        self.train_x2 = self.train_x2[train_sample_idx, :self.embedding_dim]
        self.test_x2 = self.test_x2[test_sample_idx, :self.embedding_dim]
        self.valid_labels = self.valid_labels[validation_sample_idx, :]
        self.train_labels = self.train_labels[train_sample_idx, :]
        self.test_labels = self.test_labels[test_sample_idx, :]



    def batch_generator(self, batch_size):
        l = self.train_x2.shape[0]
        for ndx in range(0, l, batch_size):
            yield (self.train_x2[ndx:min(ndx + batch_size, l), :],
                self.train_labels[ndx:min(ndx + batch_size, l),:],
                )

    def run(self, train_csv, valid_csv, test_csv, n_train_samples=400000, n_validation_samples=10000, n_test_samples = 1000, embedding_matrix=None, embedding_dim=300, max_len=50, train_x1=None, train_x2=None, save_embedding=False, save_train_data=False, contrastive=False):
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.test_csv = test_csv
        train_df, valid_df, test_df = self.import_data(train_csv, valid_csv, test_csv)
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.preprocess_word2vec(train_df, valid_df, test_df,  save_embedding, save_train_data)
        self.subsample(n_train_samples, n_validation_samples, n_test_samples)
