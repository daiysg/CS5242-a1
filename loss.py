import numpy as np

class Loss(object):
    
    def __init__(self):
        self.trainable = False # Whether there are parameters in this layer that can be trained
        self.training = False # The phrase, if for training then true

    def forward(self, inputs, targets):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, inputs, targets):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training


class SoftmaxCrossEntropy(Loss):
    def __init__(self, num_class):
        """Initialization

        # Arguments
            num_class: int, the number of category
        """
        super(SoftmaxCrossEntropy, self).__init__()
        self.num_class = num_class

    def forward(self, inputs, targets):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, num_class)
            targets: numpy array with shape (batch,)

        # Returns
            outputs: float, batch loss
            probs: numpy array with shape (batch, num_class), probability to each category with respect to each image
        """
        outputs = None
        probs = None
        #############################################################
        # code here
        # m = inputs.shape[0]
        # exps = np.exp(inputs - np.max(inputs))
        # probs = exps / np.sum(exps)
        # log_likelihood = -np.log(probs[range(m), targets])
        # outputs = np.sum(log_likelihood) / m

        #
        input_logits = inputs - np.max(inputs, axis=1, keepdims=True)
        Z = np.sum(np.exp(input_logits), axis=1, keepdims=True)
        log_probs = input_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = inputs.shape[0]
        outputs = -np.sum(log_probs[np.arange(N), targets]) / N
        #############################################################
        return outputs, probs

    def backward(self, inputs, targets):
        """Backward pass

        # Arguments
            inputs: numpy array with shape (batch, num_class), same with forward inputs
            targets: numpy array with shape (batch,), same with forward targets

        # Returns
            out_grads: numpy array with shape (batch, num_class), gradients to inputs 
        """
        #############################################################
        # code here
        # N = targets.shape[0]
        # exps = np.exp(inputs - np.max(inputs))
        # out_grads = exps / np.sum(exps)
        # out_grads[range(N), targets] -= 1
        # out_grads = out_grads / N

        out_grads = inputs.copy()
        N = inputs.shape[0]
        out_grads[np.arange(N), targets] -= 1
        out_grads /= N
        #############################################################
        return out_grads

class L2(Loss):
    def __init__(self, w=0.01):
        """Initialization

        # Arguments
            w: float, weight decay ratio.
        """
        self.w = w

    def forward(self, params):
        """Forward pass

        # Arguments
            params: dictionary, store all weights of the whole model

        # Returns
            outputs: float, L2 regularization loss
        """
        loss = 0
        for _, v in params.items():
            loss += np.sum(v**2)
        outputs = 0.5 * self.w * loss
        return outputs

    def backward(self, params):
        """Backward pass

        # Arguments
            params: dictionary, store all weights of the whole model

        # Returns
            out_grads: dictionary, gradients to each weights in params 
        """
        out_grads = {}
        for k, v in params.items():
            out_grads[k] = self.w * params[k]
        return out_grads