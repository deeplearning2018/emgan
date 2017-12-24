import pickle, gzip, os, sys, timeit, math, numpy, theano, theano.tensor as T, basic
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from gensim.models.word2vec import Word2Vec, LineSentence
from sklearn.cluster import KMeans

class GAN(object):

    def __init__(self, num_of_corpus=5, batch_size=50, seed=1):

        self.num_of_corpus = num_of_corpus
        self.batch_size = batch_size
        self.data_size = batch_size * num_of_corpus

        self.rng = numpy.random.RandomState(seed)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        self.X_train, self.X_valid, self.X_test = [], [], []
        data = pickle.load(gzip.open('time.pkl.gz', 'rb'))
        for i in range(self.num_of_corpus):
            valid_indices = np.linspace(0, data[i][0].shape[0]-1, num=data[i][0].shape[0]//8).astype('int32')
            self.X_train += [theano.shared(np.delete(data[i][0], valid_indices, 0), borrow=True)]
            self.X_valid += [theano.shared(data[i][0][valid_indices, :], borrow=True)]
            self.X_test += [theano.shared(data[i][1], borrow=True)]

        self.W_values = pickle.load(gzip.open('data/models/word2vec-%d.pkl.gz' %seed, 'rb'))
        self.W = [theano.shared(value=self.W_values[i], name=('W_%d' %i), borrow=True) for i in range(self.num_of_corpus)]
        self.W_c = theano.shared(value=self.W_values[-1], name='W_c', borrow=True)

        idf = np.load('idf.npy')
        self.idf = theano.shared(value=idf[-1,:]/idf[-1,:].max(), name='idf', borrow=True)

    def create_model(self):

        self.X = [T.imatrix(('X_%d' %i)) for i in range(self.num_of_corpus)]
        self.X_input = T.concatenate([T.tanh(T.dot(basic.row_normalize(self.X[i]*self.idf), self.W[i])) for i in range(self.num_of_corpus)], axis=0) 
        self.X_c_input = T.concatenate([T.tanh(T.dot(basic.row_normalize(self.X[i]*self.idf), self.W_c)) for i in range(self.num_of_corpus)], axis=0) 
        self.X_all_input = T.concatenate([self.X_input, self.X_c_input], axis=0)

        self.y_input = T.concatenate([i*T.ones_like(self.X[i][:,0], dtype='int32') for i in range(self.num_of_corpus)])
        self.z_input = T.concatenate([T.ones_like(self.X_input[:,0], dtype='int32'), T.zeros_like(self.X_c_input[:,0], dtype='int32')])

        self.classifier = basic.MLP(self.rng, self.X_c_input, self.y_input, n_out=self.num_of_corpus)
        self.discriminator = basic.MLP(self.rng, self.X_all_input, self.z_input, n_hidden=10, n_out=2, L2_reg=0.01)
        self.generator = [self.W_c]

    def get_cost_updates(self):

        self.Lambda = T.scalar('Lambda')
        self.mu = T.scalar('mu')
        self.lr_C = T.scalar('lrC')
        self.lr_D = T.scalar('lrD')
        self.lr_G = T.scalar('lrG')

        self.cost = self.mu*self.classifier.cost + self.Lambda*self.discriminator.cost
        g_C = [self.lr_C * T.grad(self.cost, param) for param in self.classifier.params]
        g_D = [self.lr_D * T.grad(self.cost, param) for param in self.discriminator.params]

        cost_for_generator = self.mu*self.classifier.cost - self.Lambda*self.discriminator.cost
        g_G = [self.lr_G * T.grad(cost_for_generator, param) for param in self.generator]

        params = self.classifier.params + self.discriminator.params + self.generator
        gparams = g_C + g_D + g_G
        self.updates = [(param, T.cast(param - gparam, 'float32')) for param, gparam in zip(params, gparams)]

        self.discriminator_error = T.mean(T.neq(self.discriminator.logRegressionLayer.y_pred, self.z_input))
        self.classifier_error = self.classifier.logRegressionLayer.errors()

    def test_score(self):

        test_func = theano.function(inputs=[], outputs=[self.X_c_input, self.y_input], givens={self.X[i] : self.X_test[i] for i in range(self.num_of_corpus)})
        X, y = test_func()
        kmeans = KMeans(n_clusters=self.num_of_corpus, random_state=0).fit(X)
        return basic.rand(kmeans.labels_, y)

    def test_gan(self, n_preepochs=100, n_epochs=500, save=True, output='result/cluster/gan-1.out'):

        n_train = [self.X_train[i].get_value(borrow=True).shape[0] for i in range(self.num_of_corpus)]
        n_train_batches = sum(n_train) // (self.batch_size*self.num_of_corpus)

        os = open(output, 'a', 1)
        os.write('rand index %f\n' %self.test_score())
        os.write('building...\n')

        indices = T.ivector()
        ind = range(self.batch_size)
        train_classifier = theano.function(inputs=[indices, self.Lambda, self.mu, self.lr_C, self.lr_D, self.lr_G], outputs=self.cost, updates=self.updates,
            givens={self.X[i] : self.X_train[i][indices%n_train[i]] for i in range(self.num_of_corpus)})
        discriminator_error = theano.function(inputs=[], outputs=self.discriminator_error, givens={self.X[i] : self.X_train[i] for i in range(self.num_of_corpus)})

        os.write('pretraining...\n')
        for epoch in range(n_preepochs):
            lr = 0.01 + 0.99*epoch/n_preepochs
            for minibatch_index in range(n_train_batches):
                _ = train_classifier(ind, 1.0, 1.0, lr, lr, lr)
                ind = [i + self.batch_size for i in ind]
            os.write('epoch %d, disc error %f\n' %(epoch, discriminator_error()))

        os.write('rand index %f\n' %self.test_score())
        if save:
            f = gzip.open('result/cluster/trained_w2v.pkl.gz', 'wb')
            pickle.dump(self.W_c.get_value(borrow=True), f, protocol=2)
            f.close()

        valid_error = theano.function(inputs=[], outputs=self.classifier_error, givens={self.X[i] : self.X_valid[i] for i in range(self.num_of_corpus)})
        test_error = theano.function(inputs=[], outputs=self.classifier_error, givens={self.X[i] : self.X_test[i] for i in range(self.num_of_corpus)})

        os.write('training...\n')
        min_error = np.inf
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                _ = train_classifier(ind, 0.0, 1.0, 0.1, 0.0, 0.1)
                ind = [i + self.batch_size for i in ind]
            valid_err, test_err = valid_error(), test_error()
            if valid_err < min_error + 1e-6:
                min_error, final_test_err = valid_err, test_err
            os.write('epoch %d, validation error %f, testing error %f\n' %(epoch, valid_err, test_err))

        os.write('finished with testing error %f.\n' %final_test_err)

if __name__ == '__main__':

    model = GAN()
    model.create_model()
    model.get_cost_updates()
    model.test_gan()

