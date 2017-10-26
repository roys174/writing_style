import tensorflow as tf

class Layer(object):
    def __init__(self, graph, max_len, embedding_dim):
        self.graph = graph
        self.x2 = tf.placeholder(dtype=tf.int32, shape=(None, max_len), name='x2')
        self.embedding_matrix = tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim), name='EMBED')

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

    def shortcut_biRNN(self, input, cell_type, network_dim):
        xs = self.rnn_temporal_split(input)
        # dropout = lambda y : tf.contrib.rnn.DropoutWrapper(y, output_keep_prob=0.5, seed=42)

        fw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)],
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        bw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)],
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        # fw_cells = [dropout(fw_cell(network_dim)) for fw_cell in fw_cells]
        # bw_cells = [dropout(bw_cell(network_dim)) for bw_cell in bw_cells]
        fw_cells = [fw_cell(network_dim) for fw_cell in fw_cells]
        bw_cells = [bw_cell(network_dim) for bw_cell in bw_cells]
        fw_stack = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_stack = tf.contrib.rnn.MultiRNNCell(bw_cells)
        outputs, fw_output_state, bw_output_state = tf.contrib.rnn.static_bidirectional_rnn(fw_stack,
                                                                bw_stack,
                                                                xs,
                                                                dtype=tf.float32)
        return outputs, fw_output_state, bw_output_state

    def rnn(self, input, cell_type, network_dim):
        xs = self.rnn_temporal_split(input)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse=None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse=None)}[cell_type]
        fw = fw_cell_unit(network_dim, reuse=None)
        outputs, _, _ = tf.contrib.rnn.static_rnn(fw, xs, dtype=tf.float32)
        final_state = outputs[-1]
        return final_state

    def attention(self, inputs, attention_size, time_major=False, return_alphas=False):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
         for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        """
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        inputs_shape = inputs.shape
        sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

        # Attention mechanism
        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    def dense_unit(self, input, name, input_dim, hidden_dim, output_dim):
        bn = tf.nn.batch_normalization(input, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        W1 = tf.get_variable(name="W1_"+name, shape=[input_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1_"+name, shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        h1 = tf.nn.relu(tf.matmul(bn, W1) + b1)
        d = tf.nn.dropout(h1, keep_prob = 0.1, seed = 42)
        W2 = tf.get_variable(name="W2_"+name, shape=[hidden_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="b2_"+name, shape=[output_dim], initializer=tf.contrib.layers.xavier_initializer())
        out = tf.matmul(d, W2) + b2
        bn_out = tf.nn.batch_normalization(out, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        return bn_out

class Network(Layer):
    def __init__(self, graph, max_len, embedding_dim):
        super(Network, self).__init__(graph, max_len, embedding_dim)

    def stacked_biRNN_fc_maxpool_network(self):
        embed = self.embed(self.x2)
        with tf.variable_scope("output", reuse=None) as scope:
            enc_repr, _ , _ = self.stacked_biRNN(embed, "LSTM", n_layers=3, network_dim=512)
            max_pool = tf.reduce_max(enc_repr, axis=0)
            output = self.dense_unit(max_pool, "feedforward", input_dim=1024, hidden_dim=512, output_dim=3)
        return output

    def biRNN_fc_network(self):
        embed = self.embed(self.x2)
        with tf.variable_scope("output", reuse=None) as scope:
            enc_repr, _ , _ = self.biRNN(embed, "LSTM", network_dim=512)
            output = self.dense_unit(enc_repr[-1], "feedforward", input_dim=1024, hidden_dim=512, output_dim=3)
        return output
