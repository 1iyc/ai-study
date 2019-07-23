import tensorflow as tf
import os
import sys
import argparse
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


def mnist_basic(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

    NUM_CLASSES = 10
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

    hidden1_units = 256
    hidden2_units = 256

    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope("inputs"):
            train_inputs = tf.compat.v1.placeholder(tf.float32, name="images", shape=[None, IMAGE_PIXELS])
            train_labels = tf.compat.v1.placeholder(tf.float32, name="labels", shape=[None, NUM_CLASSES])

        with tf.name_scope("hidden1"):
            weights = tf.Variable(tf.random.normal([IMAGE_PIXELS, hidden1_units], stddev=0.01), name="weights")
            hidden1 = tf.nn.relu(tf.matmul(train_inputs, weights))

            tf.compat.v1.summary.histogram("weights", weights)

        with tf.name_scope("hidden2"):
            weights = tf.Variable(tf.random.normal([hidden1_units, hidden2_units], stddev=0.01), name="weights")
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights))

            tf.compat.v1.summary.histogram("weights", weights)

        with tf.name_scope("softmax_linear"):
            weights = tf.Variable(tf.random.normal([hidden2_units, 10], stddev=0.01))
            logits = tf.matmul(hidden2, weights)

            tf.compat.v1.summary.histogram("weights", weights)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=train_labels))

            tf.compat.v1.summary.scalar("loss", loss)

        with tf.name_scope("optimizer"):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

        merged = tf.compat.v1.summary.merge_all()

        init = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver()

    epoch_size = 5
    batch_size = 100
    total_batch_size = int(mnist.train.num_examples / batch_size)

    with tf.compat.v1.Session(graph=graph) as session:
        ckpt = tf.train.get_checkpoint_state(os.path.join(log_dir))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            init.run()

        writer = tf.compat.v1.summary.FileWriter(log_dir, session.graph)

        print("Initialized")

        for epoch in xrange(epoch_size):
            total_loss = 0

            for step in xrange(session.run(global_step), session.run(global_step) + total_batch_size):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                run_metadata = tf.compat.v1.RunMetadata()

                _, summary, loss_val = session.run([train_op, merged, loss],
                                                   feed_dict={train_inputs: batch_xs, train_labels: batch_ys},
                                                   run_metadata=run_metadata)
                total_loss += loss_val

                writer.add_summary(summary, step)

            print('Epoch:', '%04d' % (epoch + 1),
                  'Avg. cost =', '{:.3f}'.format(total_loss / total_batch_size))

        writer.add_run_metadata(run_metadata, 'step%d' % session.run(global_step))

        saver.save(session, os.path.join(log_dir, 'model.ckpt'), global_step=global_step)

        print("global steps :  ", session.run(global_step))

        is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(train_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도:', session.run(accuracy,
                                  feed_dict={train_inputs: mnist.test.images,
                                             train_labels: mnist.test.labels}))

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

    mnist_basic(flags.log_dir)

    return 1


if __name__ == '__main__':
    tf.compat.v1.app.run()