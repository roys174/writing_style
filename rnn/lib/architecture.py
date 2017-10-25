import tensorflow as tf

class Layer(object):
    def __init__(self, graph, max_len, embedding_dim):
        self.graph = graph
        self.x1 = tf.placeholder(dtype=tf.int32, shape=(None, max_len), name='x1')
        self.x2 = tf.placeholder(dtype=tf.int32, shape=(None, max_len), name='x2')
        self.embedding_matrix = tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim), name='x2')

    def embed(self, input):
        return tf.nn.embedding_lookup(self.embedding_matrix, input)

    def rnn_temporal_split(self, input):
        num_steps = input.get_shape().as_list()[1]
        embed_split = tf.split(axis=1, num_or_size_splits=num_steps, value=input)
        embed_split = [tf.squeeze(x, axis=[1]) for x in embed_split]
        return embed_split

    def stacked_biRNN(self, input, cell_type, n_layers, network_dim):
        xs = self.rnn_temporal_split(input)
        dropout = lambda y : tf.contrib.rnn.DropoutWrapper(y, output_keep_prob=0.5, seed=42)

        fw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)],
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        bw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)],
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        fw_cells = [dropout(fw_cell(network_dim)) for fw_cell in fw_cells]
        bw_cells = [dropout(bw_cell(network_dim)) for bw_cell in bw_cells]
        fw_stack = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_stack = tf.contrib.rnn.MultiRNNCell(bw_cells)
        outputs, fw_output_state, bw_output_state = tf.contrib.rnn.static_bidirectional_rnn(fw_stack,
                                                                bw_stack,
                                                                xs,
                                                                dtype=tf.float32)

        return outputs, fw_output_state, bw_output_state

    def biRNN(self, input, cell_type, network_dim):
        xs = self.rnn_temporal_split(input)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse = None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse = None)}[cell_type]
        bw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse = None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse = None)}[cell_type]
        fw = fw_cell_unit(network_dim)
        bw = bw_cell_unit(network_dim)
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw,
                                                                bw,
                                                                xs,
                                                                dtype=tf.float32)
        return outputs, output_state_fw, output_state_bw

    def rnn(self, input, cell_type, network_dim):
        xs = self.rnn_temporal_split(input)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse=None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse=None)}[cell_type]
        fw = fw_cell_unit(network_dim, reuse=None)
        outputs, _, _ = tf.contrib.rnn.static_rnn(fw, xs, dtype=tf.float32)
        final_state = outputs[-1]
        return final_state

    def dense_unit(self, input, name, input_dim, hidden_dim, output_dim):
        bn = tf.nn.batch_normalization(input, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        W1 = tf.get_variable(name="W1_"+name, shape=[input_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1_"+name, shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        h1 = tf.nn.relu(tf.matmul(bn, W1) + b1)
        d = tf.nn.dropout(h1, keep_prob = 0.5, seed = 42)
        W2 = tf.get_variable(name="W2_"+name, shape=[hidden_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="b2_"+name, shape=[output_dim], initializer=tf.contrib.layers.xavier_initializer())
        out = tf.matmul(d, W2) + b2
        bn_out = tf.nn.batch_normalization(out, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        return bn_out

class Network(Layer):
    def __init__(self, graph, max_len, embedding_dim):
        super(Network, self).__init__(graph, max_len, embedding_dim)

    def biRNN_fc_network(self):
        embed = self.embed(tf.concat([self.x1, self.x2], axis=1))
        with tf.variable_scope("output", reuse=None) as scope:
            enc_repr, _ , _ = self.biRNN(embed, "LSTM", network_dim=50)
            output = self.dense_unit(enc_repr[-1], "feedforward", input_dim=50*2, hidden_dim=25, output_dim=3)
        return output
