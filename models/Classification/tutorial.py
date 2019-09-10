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

    print(x_data)
    print(y_data)

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope("inputs"):
            X = tf.placeholder(tf.float32, name="X1", shape=[None, 2])
            Y = tf.placeholder(tf.float32, name="Y1", shape=[None, 3])

        with tf.name_scope("layer1"):
            W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0), name="W1")
            b1 = tf.Variable(tf.zeros([10]), name="b1")
            L1 = tf.add(tf.matmul(X, W1), b1)
            L1 = tf.nn.relu(L1)

            tf.summary.histogram("W1", W1)
            tf.summary.histogram("b1", b1)

        with tf.name_scope("layer2"):
            W2 = tf.Variable(tf.random_uniform([10, 3], -1.0, 1.0), name="W2")
            b2 = tf.Variable(tf.zeros([3]), name="b2")

            tf.summary.histogram("W2", W2)
            tf.summary.histogram("b1", b2)

            model = tf.add(tf.matmul(L1, W2), b2)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

            tf.summary.scalar("loss", loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)

        merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    num_steps = 100000

    with tf.compat.v1.Session(graph=graph) as session:
        ckpt = tf.train.get_checkpoint_state(os.path.join(log_dir))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            init.run()

        writer = tf.summary.FileWriter(log_dir, session.graph)

        print("Initialized")

        average_loss = 0
        num_global_steps = session.run(global_step)
        for step in xrange(num_global_steps, num_global_steps + num_steps):
            feed_dict = {X: x_data, Y: y_data}

            run_metadata = tf.RunMetadata()

            _, summary, loss_val = session.run([train_op, merged, loss],
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

        saver.save(session, os.path.join(log_dir, 'model.ckpt'), global_step=global_step)

        print("global steps :  ", session.run(global_step))

        print("\n=== Test ===")
        print("X: [0, 1], Y:", session.run(tf.argmax(model, 1), feed_dict={X: [[0, 1]]}))
        print("X: [1, 0], Y:", session.run(tf.argmax(model, 1), feed_dict={X: [[1, 0]]}))

        print("model: ", session.run(tf.argmax(model, 1), feed_dict={X: x_data}))
        print("answer: :", session.run(tf.argmax(Y, 1), feed_dict={Y: y_data}))

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
