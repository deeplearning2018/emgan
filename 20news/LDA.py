import pickle, gzip, os, sys, timeit, math, numpy as np, lda, pickle, gzip, time

data = pickle.load(gzip.open('20news.pkl.gz', 'rb'))
train_set_x = []
for i in range(6):
    valid_indices = np.linspace(0, data[i][0].shape[0]-1, num=data[i][0].shape[0]//8).astype('int32')
    train_set_x += [np.delete(data[i][0], valid_indices, 0)]

phi = []
for i in range(6):
    model = lda.LDA(n_topics=50, n_iter=500, random_state=1)
    model.fit(train_set_x[i])
    phi += [model.topic_word_]
train_set_x = np.vstack(train_set_x)
model = lda.LDA(n_topics=50, n_iter=500, random_state=1)
model.fit(train_set_x)
phi += [model.topic_word_]
pickle.dump(phi, gzip.open('result/LDA/lda-model.pkl.gz', 'wb'))
