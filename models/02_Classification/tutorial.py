import tensorflow as tf
import sys
import os
import argparse
from six.moves import xrange
import numpy as np


def classification_basic(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # [털, 날개]
    x_data = np.array([[0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 0],
                       [0, 0],
                       [0, 1]])

    # 기타: 0, 포유류: 1, 조류: 2
    y_data = np.array([0, 1, 2, 0, 0, 2])

    def to_one_hot(data):
        output = np.unique(data, axis=0)
        output = output.shape[0]
        output = np.eye(output)[data]
        return output

    y_data = to_one_hot(y_data)

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope("inputs"):
            X = tf.placeholder(tf.float32, name="X1", shape=[None, 2])
            Y = tf.placeholder(tf.float32, name="Y1", shape=[None, 3])

        with tf.name_scope("weight"):
            W = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0), name="W1")
            tf.summary.histogram("W1", W)

        with tf.name_scope("bias"):
            b = tf.Variable(tf.random_uniform([3], -1.0, 1.0), name="b1")
            tf.summary.histogram("b1", b)

        with tf.name_scope("loss"):
            hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
            loss = tf.reduce_min(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
            tf.summary.scalar("loss", loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    num_steps = 100000

    with tf.compat.v1.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(log_dir, session.graph)

        init.run()
        print("Initialized")

        average_loss = 0
        for step in xrange(num_steps):
            feed_dict = {X: x_data, Y: y_data}

            run_metadata = tf.RunMetadata()

            _, summary, loss_val = session.run([optimizer, merged, loss],
                                               feed_dict=feed_dict,
                                               run_metadata=run_metadata)
            average_loss += loss_val

            writer.add_summary(summary, step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000

                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

        saver.save(session, os.path.join(log_dir, 'model.ckpt'))

        print("\n=== Test ===")
        print("X: [0, 1], Y:", session.run(tf.argmax(hypothesis), feed_dict={X: [[0, 1]]}))
        print("X: [1, 0], Y:", session.run(tf.argmax(hypothesis), feed_dict={X: [[1, 0]]}))

    writer.close()


def main(_):
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.')
    flags, unused_flags = parser.parse_known_args()

    classification_basic(flags.log_dir)

    return 1


if __name__ == '__main__':
    tf.app.run()
