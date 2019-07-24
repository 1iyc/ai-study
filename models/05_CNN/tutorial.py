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

    def nn_layer(input_tensor, filter_size, feature_num, filter_num, pooling_size, layer_name,
                 pool=tf.nn.max_pool, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([filter_size, filter_size, feature_num, filter_num])
                variable_summaries(weights)
            with tf.name_scope('conv2d'):
                conv2d = tf.nn.conv2d(input_tensor, weights, padding='SAME')
                variable_summaries(conv2d)
            with tf.name_scope('activations'):
                activations = act(conv2d, name='activations')
                tf.summary.histogram('activations', activations)
            with tf.name_scope('pooling'):
                pooled = pool(activations, ksize=[1, pooling_size, pooling_size, 1],
                               strides=[1, pooling_size, pooling_size, 1], padding='SAME')
                tf.summary.histogram('pooled', pooled)
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                tf.summary.scalar('dropout_keep_probability', keep_prob)
                dropped = tf.nn.dropout(hidden1, rate=(1 - keep_prob))
            return dropped

    hidden1 = nn_layer(x, 3, 1, 32, 2, 'layer1')
    hidden2 = nn_layer(hidden1, 3, 32, 64, 2, 'layer2')
    fc1 = nn_layer(hidden2, )

    #########
    # 신경망 모델 구성
    ######
    # 기존 모델에서는 입력 값을 28x28 하나의 차원으로 구성하였으나,
    # CNN 모델을 사용하기 위해 2차원 평면과 특성치의 형태를 갖는 구조로 만듭니다.
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # 각각의 변수와 레이어는 다음과 같은 형태로 구성됩니다.
    # W1 [3 3 1 32] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수
    # L1 Conv shape=(?, 28, 28, 32)
    #    Pool     ->(?, 14, 14, 32)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    # tf.nn.conv2d 를 이용해 한칸씩 움직이는 컨볼루션 레이어를 쉽게 만들 수 있습니다.
    # padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    # Pooling 역시 tf.nn.max_pool 을 이용하여 쉽게 구성할 수 있습니다.
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # L1 = tf.nn.dropout(L1, keep_prob)

    # L2 Conv shape=(?, 14, 14, 64)
    #    Pool     ->(?, 7, 7, 64)
    # W2 의 [3, 3, 32, 64] 에서 32 는 L1 에서 출력된 W1 의 마지막 차원, 필터의 크기 입니다.
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # L2 = tf.nn.dropout(L2, keep_prob)

    # FC 레이어: 입력값 7x7x64 -> 출력값 256
    # Full connect를 위해 직전의 Pool 사이즈인 (?, 7, 7, 64) 를 참고하여 차원을 줄여줍니다.
    #    Reshape  ->(?, 256)
    W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
    L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
    L3 = tf.matmul(L3, W3)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob)

    # 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
    W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
    model = tf.matmul(L3, W4)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    # 최적화 함수를 RMSPropOptimizer 로 바꿔서 결과를 확인해봅시다.
    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

    #########
    # 신경망 모델 학습
    ######
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(15):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)

            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: batch_xs,
                                              Y: batch_ys,
                                              keep_prob: 0.7})
            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1),
              'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

    print('최적화 완료!')

    #########
    # 결과 확인
    ######
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도:', sess.run(accuracy,
                            feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                       Y: mnist.test.labels,
                                       keep_prob: 1}))


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
