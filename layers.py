"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np
from utils.tools import *


class Layer(object):
    """
    
    """

    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False  # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        outputs = None

        #############################################################
        # code here
        N = inputs.shape[0]
        D = np.prod(inputs.shape[1:])
        x_rs = np.reshape(inputs, (N, -1))
        outputs = x_rs.dot(self.weights) + self.bias
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here

        N = inputs.shape[0]
        x_rs = np.reshape(inputs, (N, -1))
        self.b_grad = in_grads.sum(axis=0)
        self.w_grad = x_rs.T.dot(in_grads)
        dx = in_grads.dot(self.weights.T)
        out_grads = dx.reshape(inputs.shape)
        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix + ':' + self.name + '/weights': self.weights,
                prefix + ':' + self.name + '/bias': self.bias
            }
            grads = {
                prefix + ':' + self.name + '/weights': self.w_grad,
                prefix + ':' + self.name + '/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h']  # height of kernel
        self.kernel_w = conv_params['kernel_w']  # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        outputs = None

        #############################################################
        # code here

        batch, in_channel, in_height, in_width = inputs.shape
        ## add padding
        input_with_padding = np.zeros((batch, in_channel, in_height + self.pad, in_width + self.pad))
        for i in range(batch):
            for j in range(inputs[i].shape[0]):
                input_with_padding[i, j] = np.pad(inputs[i, j], int(self.pad / 2), 'constant')

        height = 1 + int((input_with_padding[0, 0].shape[0] - self.kernel_h) / self.stride)
        width = 1 + int((input_with_padding[0, 0].shape[1] - self.kernel_w) / self.stride)

        ## calcualte new X
        new_input_matrix = np.ndarray(shape=(
        input_with_padding.shape[0], self.kernel_h * self.kernel_w * input_with_padding.shape[1], height * width))

        for i in range(input_with_padding.shape[0]):
            X = []
            for k in range(height):
                for l in range(width):
                    vec = input_with_padding[i, 0:input_with_padding.shape[1],
                          k * self.stride:k * self.stride + self.kernel_h,
                          l * self.stride:l * self.stride + self.kernel_w].reshape(-1)
                    X.append(vec)
            new_input_matrix[i] = np.asarray(X).T

        ## calculate W
        W = np.array([])
        for i in range(self.out_channel):
            w_vec = self.weights[i].reshape(-1)
            if i == 0:
                W = w_vec
            else:
                W = np.vstack((W, w_vec))

        outputs = np.ndarray(shape=(inputs.shape[0], self.out_channel, height, width))

        ## calculate B
        new_bias_matrix = self.bias
        for i in range(new_input_matrix.shape[2] - 1):
            new_bias_matrix = np.vstack((new_bias_matrix, self.bias))
        new_bias_matrix = new_bias_matrix.T

        ## calculate output
        for i in range(new_input_matrix.shape[0]):
            output_w = W.dot(new_input_matrix[i])
            middle = (output_w + new_bias_matrix)
            outputs[i] = middle.reshape(self.out_channel, height, width)

        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here
        in_batch, in_channel, in_height, in_width = inputs.shape
        w_batch, w_channel, w_height, w_width = self.weights.shape

        # padding
        t_pad = int(self.pad / 2)
        x_pad = np.pad(inputs, ((0,), (0,), (t_pad,), (t_pad,)), mode='constant', constant_values=0)

        # init output
        out_height = 1 + int((in_height + self.pad - w_height) / self.stride)
        out_width = 1 + int((in_width + self.pad - w_width) / self.stride)
        out_grads = np.zeros(inputs.shape)
        out_grad_pad = np.zeros(x_pad.shape)

        out_dw = np.zeros(self.weights.shape)
        self.b_grad = np.sum(in_grads, axis=(0, 2, 3))
        for i in range(out_height):
            for j in range(out_width):
                x_pad_masked = x_pad[:, :, i * self.stride:i * self.stride + w_height,
                               j * self.stride:j * self.stride + w_width]
                for k in range(w_batch):
                    out_dw[k, :, :, :] += np.sum(x_pad_masked * (in_grads[:, k, i, j])[:, None, None, None], axis=0)
                for n in range(in_batch):
                    out_grad_pad[n, :, i * self.stride:i * self.stride + w_height,
                    j * self.stride:j * self.stride + w_width] += np.sum(
                        (self.weights[:, :, :, :] * (in_grads[n, :, i, j])[:, None, None, None]), axis=0)

        out_grads = out_grad_pad[:, :, t_pad:-t_pad, t_pad:-t_pad]

        self.w_grad = out_dw
        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix + ':' + self.name + '/weights': self.weights,
                prefix + ':' + self.name + '/bias': self.bias
            }
            grads = {
                prefix + ':' + self.name + '/weights': self.w_grad,
                prefix + ':' + self.name + '/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >= 0) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        # code here

        batch, in_channel, in_height, in_width = inputs.shape

        h_num = int((in_height - self.pool_height) / self.stride) + 1
        w_num = int((in_width - self.pool_width) / self.stride) + 1

        outputs = np.zeros((batch, in_channel, h_num, w_num))
        for i in range(batch):
            for j in range(in_channel):
                for k in range(h_num):
                    for l in range(w_num):
                        if self.pool_type == 'max':
                            outputs[i, j, k, l] = np.max(
                                inputs[i, j, k * self.stride:k * self.stride + self.pool_height,
                                l * self.stride:l * self.stride + self.pool_width])
                        elif self.pool_type == 'avg':
                            outputs[i, j, k, l] = np.average(
                                inputs[i, j, k * self.stride:k * self.stride + self.pool_height,
                                l * self.stride:l * self.stride + self.pool_width])
                            #############################################################

        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        # code here

        batch, in_channel, in_height, in_width = inputs.shape
        h_num = int((in_height - self.pool_height) / self.stride) + 1
        w_num = int((in_width - self.pool_width) / self.stride) + 1

        out_grads = np.zeros(inputs.shape)
        for i in range(batch):
            for j in range(in_channel):
                for k in range(h_num):
                    for l in range(w_num):
                        output_pool = inputs[i, j, k * self.stride:k * self.stride + self.pool_height,
                                      l * self.stride:l * self.stride + self.pool_width]
                        output_mask = None
                        if self.pool_type == 'max':
                            output_mask = output_pool == np.max(output_pool)
                        elif self.pool_type == 'avg':
                            output_mask = output_pool == np.average(output_pool)
                        out_grads[i, j, k * self.stride:k * self.stride + self.pool_height,
                        l * self.stride:l * self.stride + self.pool_width] += in_grads[i, j, k, l] * output_mask

        #############################################################
        return out_grads


class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = None
        #############################################################
        # code here
        p = self.ratio
        np.random.seed(self.seed)

        if self.training:
            ## training data set
            self.mask = (np.random.randn(*inputs.shape) < p) / p
            outputs = inputs * self.mask
        else:
            ## test dataset
            outputs = inputs

        outputs = outputs.astype(inputs.dtype, copy=False)
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = None
        #############################################################
        # code here
        if (self.training):
            out_grads = in_grads * self.mask
        else:
            out_grads = in_grads
        #############################################################
        return out_grads


class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
