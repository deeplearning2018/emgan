import pickle, gzip, os, sys, timeit, math, numpy, theano, theano.tensor as T, basic
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from gensim.models.word2vec import Word2Vec, LineSentence

class Generator(object):

    def __init__(self, rng, theano_rng, batch_size, num_of_corpus, n_noise=100, n_hidden=50, n_hidden_per_class=50, n_output=5000):

        lda_phi = pickle.load(gzip.open('result/LDA/lda-model.pkl.gz', 'rb'))
        W_h0 = theano.shared(value=rng.uniform(low=-0.01, high=0.01, size=(n_noise, n_hidden)).astype('float32'), name='Wh0', borrow=True)
        W_hm = [theano.shared(value=rng.uniform(low=-0.01, high=0.01, size=(n_noise, n_hidden_per_class)).astype('float32'), name='Whm', borrow=True) for i in range(num_of_corpus)]
        W_o0 = theano.shared(value=np.log(lda_phi[-1]).astype('float32'), name='Wo0', borrow=True)
        W_om = [theano.shared(value=np.log(lda_phi[i]).astype('float32'), name='Wom', borrow=True) for i in range(num_of_corpus)]

        self.noise = theano_rng.uniform(size=(batch_size*num_of_corpus, n_noise), low=-1., high=1.)
        self.batch_size_var = self.noise.shape[0]//num_of_corpus
        self.output = T.concatenate([T.nnet.softmax(T.dot(T.tanh(T.dot(self.noise[i*self.batch_size_var:(i+1)*self.batch_size_var], W_h0)), W_o0)
            + T.dot(T.tanh(T.dot(self.noise[i*self.batch_size_var:(i+1)*self.batch_size_var], W_hm[i])), W_om[i])) for i in range(num_of_corpus)], axis=0)
        self.params = [W_h0] + W_hm + [W_o0] + W_om

class GAN(object):

    def __init__(self, num_of_corpus=5, batch_size=50, n_noise=100, seed=1):

        self.seed = seed
        self.num_of_corpus = num_of_corpus
        self.batch_size = batch_size
        self.n_noise = n_noise
        self.rng = numpy.random.RandomState(seed)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        self.X_train, self.X_valid, self.X_test = [], [], []
        data = pickle.load(gzip.open('time.pkl.gz', 'rb'))
        for i in range(self.num_of_corpus):
            valid_indices = np.linspace(0, data[i][0].shape[0]-1, num=data[i][0].shape[0]//8).astype('int32')
            self.X_train += [theano.shared(np.delete(data[i][0], valid_indices, 0).astype('int32'), borrow=True)]
            self.X_valid += [theano.shared(data[i][0][valid_indices, :].astype('int32'), borrow=True)]
            self.X_test += [theano.shared(data[i][1].astype('int32'), borrow=True)]

        self.W_values = pickle.load(gzip.open('data/models/word2vec-%d.pkl.gz' %seed, 'rb'))
        self.W = theano.shared(value=self.W_values[-1], name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(self.W_values[-1].shape[1]).astype('float32'), name='b', borrow=True)

        idf = np.load('idf.npy')
        self.idf = theano.shared(value=idf[-1,:]/idf[-1,:].max(), name='idf', borrow=True)

    def create_model(self):

        self.X = [T.imatrix(('X_%d' %i)) for i in range(self.num_of_corpus)]
        self.X_input = T.concatenate([(T.dot(basic.row_normalize(self.X[i]*self.idf), self.W) + self.b) for i in range(self.num_of_corpus)], axis=0) 
        self.generator = Generator(self.rng, self.theano_rng, self.batch_size, self.num_of_corpus)
        self.X_all_input = T.concatenate([self.X_input, (T.dot(self.generator.output, self.W) + self.b)], axis=0)

        self.y_input = T.concatenate([i*T.ones_like(self.X[i][:,0], dtype='int32') for i in range(self.num_of_corpus)])
        self.g_label_input = T.concatenate([(i + self.num_of_corpus)*T.ones_like(self.generator.output[:self.generator.batch_size_var, 0], dtype='int32') for i in range(self.num_of_corpus)])
        self.y_all_input = T.concatenate([self.y_input, self.g_label_input])

        self.discriminator = basic.MLP(self.rng, self.X_all_input, self.y_all_input, n_out=2*self.num_of_corpus) 
        self.classifier = basic.MLP(self.rng, self.X_input, self.y_input, n_out=self.num_of_corpus) 

    def get_cost_updates(self):

        self.lr_D = T.scalar('lrD')
        self.lr_G = T.scalar('lrG')

        self.discriminator_cost = self.discriminator.logRegressionLayer.negative_log_likelihood()
        discriminator_params = [self.W, self.b] + self.discriminator.params
        g_D = [self.lr_D * T.grad(self.discriminator_cost, param) for param in discriminator_params]

        self.classification_error = T.mean(T.neq(self.discriminator.logRegressionLayer.y_pred, self.y_all_input)[:T.shape(self.X_input)[0]])
        self.gen_classification_error = T.mean(T.neq(self.discriminator.logRegressionLayer.y_pred, self.y_all_input)[T.shape(self.X_input)[0]:])
        self.discrimination_error = T.mean(T.neq(self.discriminator.logRegressionLayer.y_pred//self.num_of_corpus, self.y_all_input//self.num_of_corpus))

        self.cost_per_gen, updates = theano.scan(fn=lambda p, y: T.log(p[y]/(p[y] + p[y-self.num_of_corpus])), outputs_info=None,
            sequences=[self.discriminator.logRegressionLayer.p_y_given_x[T.shape(self.X_input)[0]:], self.g_label_input], non_sequences=None)
        self.generator_cost = T.mean(self.cost_per_gen)
        g_G = [self.lr_G * T.grad(self.generator_cost, param) for param in self.generator.params]

        params = discriminator_params + self.generator.params
        gparams = g_D + g_G
        self.updates = updates + [(param, T.cast(param - gparam, 'float32')) for param, gparam in zip(params, gparams)]

        self.lr_C = T.scalar('lrC')
        self.classifier_cost = self.classifier.logRegressionLayer.negative_log_likelihood()
        self.classifier_error = self.classifier.logRegressionLayer.errors()
        classifier_params = [self.W, self.b] + self.classifier.params
        g_C = [self.lr_C * T.grad(self.classifier.logRegressionLayer.negative_log_likelihood(), param) for param in classifier_params]
        self.c_updates = [(param, T.cast(param - gparam, 'float32')) for param, gparam in zip(classifier_params, g_C)]

    def post_process(self, data, n_words=20, n_docs=50, categories = ['Entertainment', 'Ideas', 'Politics', 'US', 'World'], stemming_dict='data/stemming_dict.pkl.gz', 
        stopwords='data/stopwords.pkl.gz', filename='result/gan/test_data.txt'):

        unstemmer = pickle.load(gzip.open(stemming_dict, 'rb'))
        stop_words = pickle.load(gzip.open(stopwords, 'rb'))
        vocab = [line.strip() for line in open('data/vocab.txt', 'r')]
        os = open(filename, 'w')

        for i in range(self.num_of_corpus):
            os.write('category \"%s\":\n' %categories[i])
            category = data[i*data.shape[0]//self.num_of_corpus:(i+1)*data.shape[0]//self.num_of_corpus, :]
            for j in range(min(category.shape[0], n_docs)):
                ind = np.argsort(category[j,:])[::-1]
                words = [unstemmer[vocab[k]] if vocab[k] in unstemmer else vocab[k] for k in ind]
                words = [v for v in words if v not in stop_words]
                os.write(' '.join(words[:n_words]) + '\n')

        os.close()

    def test_gan(self, n_epochs=500, n_post_epochs=500, save=True, output='result/gan/gan-1.out'):

        n_train = [self.X_train[i].get_value(borrow=True).shape[0] for i in range(self.num_of_corpus)]
        n_train_batches = sum(n_train) // (self.batch_size*self.num_of_corpus)
        test_batch_size = sum([self.X_test[i].get_value(borrow=True).shape[0] for i in range(self.num_of_corpus)]) // self.num_of_corpus

        os = open(output, 'w', 1)
        os.write('building...\n')

        x = T.matrix()
        indices = T.ivector()
        train_model = theano.function(inputs=[indices, self.lr_D, self.lr_G], outputs=self.discriminator_cost, updates=self.updates,
            givens={self.X[i] : self.X_train[i][indices%n_train[i]] for i in range(self.num_of_corpus)})

        test_model_givens = {self.X[i] : self.X_test[i] for i in range(self.num_of_corpus)}
        test_model_givens[self.generator.noise] = self.theano_rng.uniform(size=(test_batch_size, self.n_noise), low=-1., high=1.)
        test_model = theano.function(inputs=[], outputs=[self.classification_error, self.discrimination_error, self.gen_classification_error], givens=test_model_givens)

        ind = range(self.batch_size)
        os.write('training...\n')

        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                _ = train_model(ind, 0.1, 0.001)
                ind = [i + self.batch_size for i in ind]
            c_err, d_err, g_err = test_model()
            os.write(('epoch %i, classification error %f, discrimination error %f, gan classification error %f\n') %(epoch, c_err, d_err, g_err))

        if save:
            test_data = theano.function(inputs=[], outputs=self.generator.output)
            self.post_process(test_data())  
            f = gzip.open('result/gan/output.pkl.gz', 'wb')
            pickle.dump(np.vstack([test_data() for i in range(4000//self.batch_size)]), f, protocol=2)
            f.close()

        self.classifier.hiddenLayer.W.set_value(self.discriminator.hiddenLayer.W.get_value(borrow=True), borrow=True)
        self.classifier.hiddenLayer.b.set_value(self.discriminator.hiddenLayer.b.get_value(borrow=True), borrow=True)
        self.classifier.logRegressionLayer.W.set_value(self.discriminator.logRegressionLayer.W.get_value(borrow=True)[:,:self.num_of_corpus], borrow=True)
        self.classifier.logRegressionLayer.b.set_value(self.discriminator.logRegressionLayer.b.get_value(borrow=True)[:self.num_of_corpus], borrow=True)

        train_classifier = theano.function(inputs=[indices, self.lr_C], outputs=self.classifier_cost, updates=self.c_updates,
            givens={self.X[i] : self.X_train[i][indices%n_train[i]] for i in range(self.num_of_corpus)})

        valid_error = theano.function(inputs=[], outputs=self.classifier_error, givens={self.X[i] : self.X_valid[i] for i in range(self.num_of_corpus)})
        test_error = theano.function(inputs=[], outputs=self.classifier_error, givens={self.X[i] : self.X_test[i] for i in range(self.num_of_corpus)})

        os.write('testing...\n')
        min_error = np.inf
        for epoch in range(n_post_epochs):
            for minibatch_index in range(n_train_batches):
                _ = train_classifier(ind, 0.1)
                ind = [i + self.batch_size for i in ind]
            valid_err, test_err = valid_error(), test_error()
            if valid_err < min_error + 1e-6:
                min_error, final_test_err = valid_err, test_err
            os.write('epoch %d, validation error %f, testing error %f\n' %(epoch, valid_err, test_err))

        os.write('finished with testing error %f.\n' %final_test_err)

if __name__ == '__main__':

    dmgan = GAN()
    dmgan.create_model()
    dmgan.get_cost_updates()
    dmgan.test_gan()

