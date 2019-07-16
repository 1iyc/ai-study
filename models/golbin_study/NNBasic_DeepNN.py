import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
print("x_data.shape", x_data.shape)

# [기타, 포유류, 조류]
y_data = np.array([
    [0],  # 기타
    [1],  # 포유류
    [2],  # 조류
    [0],
    [0],
    [2]
])
print("y_data.shape", y_data.shape)

nb_class = 3
print("nb_class", nb_class)

y_one_hot = tf.one_hot(y_data, nb_class)
y_one_hot = tf.reshape(y_one_hot, [-1, nb_class])
print("y_one_hot", y_one_hot)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, nb_class], -1., 1.))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([nb_class]))

L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=model))

optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.3)
train_op = optimizer.minimize(cost)

prediction = tf.argmax(model, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        sess.run(train_op, feed_dict={X: x_data})

        if (step + 1) % 10 == 0:
            print(step + 1, sess.run([cost, accuracy], feed_dict={X: x_data}))

    print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
    print('실제값:', y_data.flatten())

