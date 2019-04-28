from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(tf.random_normal([2, 2]), name='weight')
        print("W: ", self.W)
        self.B = tf.Variable(tf.zeros([2]), name='bias')
        print("B: ", self.B)

    def call(self, inputs):
        return tf.matmul(inputs, self.W) + self.B


training_inputs = tf.constant([[0, 0],
                               [0, 1],
                               [1, 0],
                               [1, 1]], dtype='float32')
print(training_inputs)

training_outputs = tf.constant([[1, 0],  # 0
                                [1, 0],  # 0
                                [1, 0],  # 0
                                [0, 1]], dtype='float32')  # 1
print(training_outputs)


# The loss function to be optimized
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])


# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(3000):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                              global_step=tf.train.get_or_create_global_step())
    if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

correct_prediction = tf.equal(tf.floor(model(training_inputs) + 0.5), training_outputs)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuracy: ", accuracy)
