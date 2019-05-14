from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        initializer = tf.contrib.layers.xavier_initializer()

        self.W1 = tf.Variable(initializer([2, 2]), name='weight1')
        print("W1: ", self.W1)
        self.B1 = tf.Variable(tf.zeros([2]), name='bias1')
        print("B1: ", self.B1)

        self.W2 = tf.Variable(initializer([2, 1]), name='weight2')
        print("W2: ", self.W2)
        self.B2 = tf.Variable(tf.zeros([1]), name='bias2')
        print("B2: ", self.B2)

    def call(self, inputs):
        L1 = tf.sigmoid(tf.matmul(inputs, self.W1) + self.B1)
        return tf.matmul(L1, self.W2) + self.B2


training_inputs = tf.constant([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]], dtype='float32')
print(training_inputs)

training_outputs = tf.constant([[0],
                                [1],
                                [1],
                                [0]], dtype='float32')
print(training_outputs)


# The loss function to be optimized
def loss(model, inputs, targets):
    # print("inputs", inputs)
    # print("targets", targets)
    # cost = -tf.reduce_mean(targets*tf.log(model(inputs)) + (1-targets)*tf.log(1-model(inputs)))
    # print("cost1", cost)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model(inputs), labels=targets))
    # print("cost2", cost)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model(inputs), labels=targets))
    return cost


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W1, model.B1, model.W2, model.B2])


# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
optimizer = tf.train.AdamOptimizer(learning_rate=0.3)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(300000):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W1, model.B1, model.W2, model.B2]),
                              global_step=tf.train.get_or_create_global_step())
    if i % 100 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W1 = {}, B1 = {}".format(model.W1.numpy(), model.B1.numpy()))
print("W2 = {}, B2 = {}".format(model.W2.numpy(), model.B2.numpy()))

correct_prediction = tf.equal(tf.floor(model(training_inputs) + 0.5), training_outputs)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuracy: ", accuracy)

print("0, 0", model([[0., 0.]]))
print("0, 1", model([[0., 1.]]))
print("1, 0", model([[1., 0.]]))
print("1, 1", model([[1., 1.]]))
