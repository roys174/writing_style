import tensorflow as tf


class SoftmaxLoss(object):

    def __init__(self):
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None, 3), name='labels')

    def cross_entropy(self, logits):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.labels)
        loss = tf.reduce_mean(losses, name="loss")
        tf.summary.scalar('cross_entropy_loss', loss)
        return loss


class SigmoidLoss(object):

    def __init__(self):
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None, 3), name='labels')

    def cross_entropy(self, logits):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = tf.to_float(self.labels))
        loss = tf.reduce_mean(losses, name="loss")
        tf.summary.scalar("cross_entropy_loss", loss)
        return loss

class Accuracy(object):

    def softmax_accuracy(self, labels, output):
        pred = tf.nn.softmax(output)
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        training_summary = tf.summary.scalar("training_accuracy", accuracy)
        validation_summary = tf.summary.scalar("validation_accuracy", accuracy)
        return accuracy, training_summary, validation_summary

    def sigmoid_accuracy(self, labels, output):
        pred = tf.sigmoid(output)
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        training_summary = tf.summary.scalar("training_accuracy", accuracy)
        validation_summary = tf.summary.scalar("validation_accuracy", accuracy)
        return accuracy, training_summary, validation_summary

    def distance_accuracy(self, labels, output):
        pred = tf.to_int32(tf.where(tf.greater_equal(output, tf.constant(0.5))))
        correct_predictions = tf.equal(pred, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        training_summary = tf.summary.scalar("training_accuracy", accuracy)
        validation_summary = tf.summary.scalar("validation_accuracy", accuracy)
        return accuracy, training_summary, validation_summary


class Optimization(object):

    def adam(self, loss, lr):
        return tf.train.AdamOptimizer(lr).minimize(loss)
