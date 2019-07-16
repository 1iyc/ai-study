import tensorflow as tf
import sys
import os
import argparse
from six.moves import xrange
from tensorflow.contrib.tensorboard.plugins import projector


def linear_regression_basic(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    x_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    y_data = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope("inputs"):
            X = tf.placeholder(tf.float32, name="X1")
            Y = tf.placeholder(tf.float32, name="Y1")

        with tf.name_scope("weight"):
            W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="W1")
            tf.summary.histogram("W1", W)

        with tf.name_scope("bias"):
            b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="b1")
            tf.summary.histogram("b1", b)

        with tf.name_scope("loss"):
            hypothesis = W * X + b
            loss = tf.reduce_min(tf.square(hypothesis - Y))
            tf.summary.scalar("loss", loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=3e-5).minimize(loss)

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

            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000

                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

        saver.save(session, os.path.join(log_dir, 'model.ckpt'))

        print("\n=== Test ===")
        print("X: 35, Y:", session.run(hypothesis, feed_dict={X: 35}))
        print("X: 100, Y:", session.run(hypothesis, feed_dict={X: 100}))

        # config = projector.ProjectorConfig()
        # embedding_conf = config.embeddings.add()
        # embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
        # projector.visualize_embeddings(writer, config)

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

    linear_regression_basic(flags.log_dir)

    return 1


if __name__ == '__main__':
    tf.app.run()