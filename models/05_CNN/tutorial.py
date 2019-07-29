# 이미지 처리 분야에서 가장 유명한 신경망 모델인 CNN 을 이용하여 더 높은 인식률을 만들어봅니다.
import tensorflow as tf
import sys
import os
import argparse

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
    # Import data
    mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

    sess = tf.InteractiveSession()

    # Input placeholder
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        tf.summary.image('input', x, 10)
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)

    # with tf.name_scope('input_reshape'):
    #     image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    #     tf.summary.image('input', image_shaped_input, 10)

    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=00.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def conv_layer(input_tensor, filter_size, feature_num, filter_num, pooling_size, layer_name,
                 pool=tf.nn.max_pool, act=tf.nn.relu):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('filters'):
                filters = weight_variable([filter_size, filter_size, feature_num, filter_num])
                variable_summaries(filters)
            with tf.name_scope('conv2d'):
                conv2d = tf.nn.conv2d(input_tensor, filters, padding='SAME')
                tf.summary.histogram('conv2d', conv2d)
            with tf.name_scope('activations'):
                activations = act(conv2d, name='activations')
                tf.summary.histogram('activations', activations)
            with tf.name_scope('pooling'):
                pooled = pool(activations, ksize=[1, pooling_size, pooling_size, 1],
                               strides=[1, pooling_size, pooling_size, 1], padding='SAME')
                tf.summary.histogram('pooled', pooled)
            with tf.name_scope('dropout'):
                dropped = tf.nn.dropout(pooled, rate=(1 - keep_prob))
            return dropped

    conv1 = conv_layer(x, 3, 1, 32, 2, 'hidden_layer1')
    conv2 = conv_layer(conv1, 3, 32, 64, 2, 'hidden_layer2')

    def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('reshape'):
                reshaped = tf.reshape(input_tensor, [-1, input_dim])
                tf.summary.histogram('reshaped', reshaped)
            with tf.name_scope('Wx'):
                preactivate = tf.matmul(reshaped, weights)
                tf.summary.histogram('pre_activations', preactivate)
            with tf.name_scope('activations'):
                activations = act(preactivate, name='activations')
                tf.summary.histogram('activations', activations)
            with tf.name_scope('dropout'):
                dropped = tf.nn.dropout(activations, rate=(1 - keep_prob))
            return dropped

    fc1 = fc_layer(conv2, 7 * 7 * 64, 256, 'fc_layer1')

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('Wx'):
                preactivate = tf.matmul(input_tensor, weights)
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    model = nn_layer(fc1, 256, 10, 'model')

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_, logits=model)
            tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(3e-5).minimize(cross_entropy)
        # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            # xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            xs, ys = mnist.train.next_batch(100)
            xs = xs.reshape(-1, 28, 28, 1)
            k = 0.9
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            xs = xs.reshape(-1, 28, 28, 1)
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(1000):
        if i % 10 == 11:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    with tf.Graph().as_default():
        train()


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.')
    FLAGS, unparsed = parser.parse_known_args()

    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
