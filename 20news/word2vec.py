import gensim, os, re, nltk, numpy as np
from gensim.models.word2vec import Word2Vec, LineSentence

categories = ['religion', 'computer', 'cars', 'sport', 'science', 'politics']
data = ['train', 'test']
for d in data:
    w = dict()
    for category in categories:
        w[category] = open('data/cleaned/cleaned_%s_%s.txt' %(category, d), 'w')
    for line in open('data/20ng-%s-stemmed.txt' %d, 'r'):
        columns = line.strip().split('\t')
        if columns[0] == 'alt.atheism' or 'religion' in columns[0]:
            type = 'religion' 
        elif 'comp' in columns[0]:
            type = 'computer'
        elif columns[0] == 'rec.autos' or columns[0] == 'rec.motorcycles':
            type = 'cars'
        elif 'rec.sport' in columns[0]:
            type = 'sport'
        elif 'sci' in columns[0]:
            type = 'science'
        elif 'politics' in columns[0]:
            type = 'politics'
        else:
            type = 'misc'
        if type != 'misc' and len(columns) == 2 and columns[1] != '':
            w[type].write(columns[1] + '\n')
    for category in categories:
        w[category].close()

randomized_models = []
seeds = [1, 123, 888, 1234, 8888]
for i in range(5):
    models = dict()
    for category in categories:
        models[category] = Word2Vec(LineSentence('data/cleaned/cleaned_%s_train.txt' %category), seed=seeds[i], size=300, window=5, min_count=5, workers=4)
    # os.system('cat ' + ' '.join([('data/cleaned/cleaned_%s_train.txt' %category) for category in categories]) + ' > data/cleaned/cleaned_all_train.txt')
    models['all'] = Word2Vec(LineSentence('data/cleaned/cleaned_all_train.txt'), size=300, window=5, min_count=5, workers=4)
    randomized_models += [models]

vocab = set(randomized_models[0]['all'].vocab.keys())
for category in categories:
    vocab = vocab.union(randomized_models[0][category].vocab.keys())

vocab = list(vocab)
indices_of_vocab = dict({vocab[i] : i for i in range(len(vocab))})
count_of_vocab = np.zeros(len(vocab), dtype='int32')
for category in categories:
    lines = open('data/cleaned/cleaned_'+category+'_train.txt', 'r').readlines()
    for line in lines:
        for w in line.strip().split(' '):
            if w in indices_of_vocab:
                count_of_vocab[indices_of_vocab[w]] += 1

vocab = [vocab[i] for i in np.argsort(count_of_vocab)[::-1][:5000]]
indices_of_vocab = dict({vocab[i] : i for i in range(len(vocab))})
w = open('data/vocab.txt', 'w')
for v in vocab:
    w.write(v+'\n')

w.close()

DF = np.zeros((6, len(vocab)), dtype='float32')
N = [0]*6
for i in range(6):
    cleaned_docs = open('data/cleaned/cleaned_'+categories[i]+'_train.txt', 'r').readlines()
    N[i] = len(cleaned_docs)
    df = np.zeros(len(vocab), dtype='int32')
    for line in cleaned_docs:
        df_per_doc = np.zeros(len(vocab), dtype='int32')
        sentences = nltk.sent_tokenize(line.strip())
        for sent in sentences:
            for w in (re.sub('\.+$', '', sent)).split(' '):
                if w in indices_of_vocab:
                    df_per_doc[indices_of_vocab[w]] += 1
        df += np.sign(df_per_doc)
    DF[i,:] = df

idf = np.zeros((7, len(vocab)), dtype='float32')
for i in range(6):
    idf[i,:] = np.log(N[i]/np.maximum(1., DF[i,:]))

idf[6,:] = np.log(sum(N)/np.maximum(1., DF.sum(axis=0)))
np.save('idf.npy', idf)

X = []
for i in range(6):
    X_i_train, X_i_test = [], []
    cleaned_train = open('data/cleaned/cleaned_'+categories[i]+'_train.txt', 'r').readlines()
    cleaned_test = open('data/cleaned/cleaned_'+categories[i]+'_test.txt', 'r').readlines()
    for line in cleaned_train:
        tf_per_doc = [0]*len(vocab)
        sentences = nltk.sent_tokenize(line.strip())
        for sent in sentences:
            for w in (re.sub('\.+$', '', sent)).split(' '):
                if w in indices_of_vocab:
                    tf_per_doc[indices_of_vocab[w]] += 1
        if sum(tf_per_doc) > 0:
            X_i_train += [tf_per_doc]
    for line in cleaned_test:
        tf_per_doc = [0]*len(vocab)
        sentences = nltk.sent_tokenize(line.strip())
        for sent in sentences:
            for w in (re.sub('\.+$', '', sent)).split(' '):
                if w in indices_of_vocab:
                    tf_per_doc[indices_of_vocab[w]] += 1
        if sum(tf_per_doc) > 0:
            X_i_test += [tf_per_doc]
    X += [[np.array(X_i_train).astype('int32'), np.array(X_i_test).astype('int32')]]

import pickle, gzip
f = gzip.open('20news.pkl.gz', 'wb')
pickle.dump(X, f, protocol=2)
f.close()

categories += ['all']
vocab = [line.strip() for line in open('data/vocab.txt', 'r')]
for j in range(5):
    W = []
    for i in range(7):
        W_i = []
        model = randomized_models[j][categories[i]]
        for k in range(len(vocab)):
            W_i += [model[vocab[k]]] if vocab[k] in model else [np.zeros(300)]
        W += [np.array(W_i).astype('float32')]
    f = gzip.open('data/models/word2vec-%s.pkl.gz' %seeds[j], 'wb')
    pickle.dump(W, f, protocol=2)
    f.close()
