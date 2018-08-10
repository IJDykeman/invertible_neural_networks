import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import uuid



def split(x):
    half_x_dim = x.shape[-1] // 2
    return x[..., :half_x_dim], x[..., half_x_dim:]

def fuse(x1, x2):
    return tf.concat([x1, x2], -1)


class HalfChannelsRotator:
    def rotate(self, x, unique_name):
        """
        This method could be replaced with shuffling (as in Real NVP) or with a learned rotation, as in GLOW.
        This rotation by reversal of feature order is used in NICE
        """
        return x[...,::-1]

    def rotate_inverse(self, x, unique_name):
        return x[...,::-1]

class DenseLayer:
    def __init__(self, x, rotator, unique_name=""):
        # we need to ensure that someone does not accidentalyl end up with 2 layers with the same name
        self.rotator = rotator
        self.unique_name = unique_name + uuid.uuid4().hex.upper()[0:6] 
        half_x_dim = x.get_shape()[-1].value // 2
        x_rotated = self.rotator.rotate(x, self.unique_name)
        x1, x2 = split(x_rotated)
        scale, translate = self._get_transform(x1)
        x2_transformed = (x2 * scale) + translate
        self.forward_output = fuse(x1, x2_transformed)
        
        # real NVP eq 6 .  Omitting the constant part of the determinant that is
        # the sum of the ones on the trace of the upper left of the matrix.
        self.determinant = tf.reduce_sum(scale, axis=-1)

    def _get_transform(self, x):
        input_dim = x.get_shape()[1].value
        with tf.variable_scope(self.unique_name, reuse=tf.AUTO_REUSE):
            h1 = slim.fully_connected(x, input_dim + 5, activation_fn=tf.nn.elu) # +5 just to show that these can be any size
            h2 = slim.fully_connected(h1, input_dim + 5, activation_fn=tf.nn.elu) # +5 just to show that these can be any size
            h3 = slim.fully_connected(h2, input_dim, activation_fn=tf.nn.elu)
            scale = tf.exp(slim.fully_connected(h3, input_dim, activation_fn=tf.nn.elu))
            translate = slim.fully_connected(h3, input_dim)
        return scale, translate

    def reverse(self, y):
        x1, x2_transformed = split(y)
        scale, translate = self._get_transform(x1)
        x2 = (x2_transformed - translate) / scale
        x_rotated = tf.concat([x1, x2], -1)
        x = self.rotator.rotate_inverse(x_rotated, self.unique_name)
        self.reverse_output = x


class ConvolutionalLayer:
    def __init__(self, x, rotator, unique_name=""):
        # we need to ensure that someone does not accidentalyl end up with 2 layers with the same name
        self.rotator = rotator
        self.unique_name = unique_name + uuid.uuid4().hex.upper()[0:6] 
        half_x_dim = x.get_shape()[-1].value // 2
        x_rotated = self.rotator.rotate(x, self.unique_name)
        x1, x2 = split(x_rotated)
        scale, translate = self._get_transform(x1)
        x2_transformed = (x2 * scale) + translate
        self.forward_output = fuse(x1, x2_transformed)
        
        # real NVP eq 6 .  Omitting the constant part of the determinant that is
        # the sum of the ones on the trace of the upper left of the matrix.
        self.determinant = tf.reduce_sum(scale, axis=[1,2,3])

    def _get_transform(self, x):
        input_dim = x.get_shape()[-1].value
        with tf.variable_scope(self.unique_name, reuse=tf.AUTO_REUSE):
            h1 = slim.conv2d(x, 32, [3, 3], activation_fn=tf.nn.elu) # +5 just to show that these can be any size
            scale = tf.exp(slim.conv2d(h1, input_dim, [3, 3], activation_fn=tf.nn.elu, padding='SAME'))
            translate = slim.conv2d(h1, input_dim, [3, 3], activation_fn=tf.nn.elu, padding='SAME')
        return scale, translate

    def reverse(self, y):
        x1, x2_transformed = split(y)
        scale, translate = self._get_transform(x1)
        x2 = (x2_transformed - translate) / scale
        x_rotated = tf.concat([x1, x2], -1)
        x = self.rotator.rotate_inverse(x_rotated, self.unique_name)
        self.reverse_output = x


# for i in range(10):
#     x = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
#     layer = DenseLayer(x, HalfChannelsRotator(), unique_name="layer1")
#     y = layer.forward_output
#     layer.reverse(y)
#     x_recovered = layer.reverse_output
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     # show that the inverse is correct
#     assert np.allclose(x.eval(session=sess), x_recovered.eval(session=sess))
#     print i

for i in range(10):
    initial_input_tensor = tf.placeholder(shape=[5,20,20,4],dtype = tf.float32)
    layer_input_tensor=initial_input_tensor
    layers = []
    n_layers = 5
    for j in range(n_layers):
        layer = ConvolutionalLayer(layer_input_tensor, HalfChannelsRotator(), unique_name="layer1")
        y = layer.forward_output
        layer_input_tensor = y
        layers.append(layer)
    final_output = y
    reverse_value = final_output
    for j in range(n_layers):
        layers[-j].reverse(reverse_value)
        reverse_value = layers[-j].reverse_output

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # # show that the inverse is correct
    # # assert np.allclose(initial_input_tensor.eval(session=sess), net_input_recovered.eval(session=sess))
    # # z = net_output.eval(session=sess)

    x = np.random.rand(*[5,20,20,4]) * 1.3
    z = final_output.eval(session=sess, feed_dict={initial_input_tensor:x})
    net_input_recovered = reverse_value.eval(session=sess, feed_dict={final_output:z})
    # # print z
    # assert np.allclose(x, net_input_recovered.eval(session=sess))
    # print i
