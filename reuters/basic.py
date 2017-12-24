import pickle, gzip, os, sys, timeit, math, numpy, theano, theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nnet.bn import batch_normalization

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, y):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input
        self.y = y

    def negative_log_likelihood(self):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])

    def errors(self):
        return T.mean(T.neq(self.y_pred, self.y))

    def get_cost_updates(self, lr):
        cost = self.negative_log_likelihood()
        g_W = T.grad(cost=cost, wrt=self.W)
        g_b = T.grad(cost=cost, wrt=self.b)
        updates = [(self.W, self.W - lr * g_W), (self.b, self.b - lr * g_b)]
        return cost, updates

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh, bn=False):

        self.input = input
        if W is None:
            W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W, self.b = W, b
        lin_output = T.dot(input, self.W) + self.b
        if bn:
            self.gamma = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), name='gamma')
            self.beta = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX), name='beta')
            mean = lin_output.mean(0, keepdims=True)
            std = T.sqrt(lin_output.std(0, keepdims=True)**2 + 0.01)
            output = batch_normalization(inputs=lin_output, gamma=self.gamma, beta=self.beta, mean=mean, std=std)
        else:
            output = lin_output

        self.output = (output if activation is None else activation(output))
        # parameters of the model
        self.params = [self.W, self.b, self.gamma, self.beta] if bn else [self.W, self.b]

class MLP(object):

    def __init__(self, rng, input, y, n_in=300, n_hidden=50, n_out=3, L1_reg=0.0001, L2_reg=0.0001, bn=False):

        self.rng = rng
        self.input = input
        self.hiddenLayer = HiddenLayer(rng=self.rng, input=self.input, n_in=n_in, n_out=n_hidden, activation=T.nnet.sigmoid, bn=bn)
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out, y=y)
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()
        self.cost = self.logRegressionLayer.negative_log_likelihood() + L2_reg*self.L2_sqr + L1_reg*self.L1
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def row_normalize(m):

    return m / m.sum(axis=1).reshape((m.shape[0], 1))

def rand(x, y):

    same = 0
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[0]):
            if (x[i]==x[j] and y[i]==y[j]) or (x[i]!=x[j] and y[i]!=y[j]):
                same += 1
    return same/(x.shape[0]*(x.shape[0]-1)/2.)

