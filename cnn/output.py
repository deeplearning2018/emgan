import gensim, os, re, nltk, numpy as np, pickle, gzip, string
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

stemming_dict = pickle.load(gzip.open('data/stemming_dict.pkl.gz'))
vocab = [v.strip() for v in open('data/vocab.txt', 'r')]
vocab = [(stemming_dict[v] if v in stemming_dict else v) for v in vocab]
indices_of_vocab = {vocab[i] : i for i in range(len(vocab))}

W = normalize(pickle.load(gzip.open('data/models/word2vec-1.pkl.gz', 'rb'))[-1])
W_new = normalize(pickle.load(gzip.open('result/cluster/trained_w2v.pkl.gz')))
W_sim = W.dot(W.T)
W_new_sim = W_new.dot(W_new.T)

diff = []
for i in range(len(vocab)):
    diff += [len(set(np.argsort(W_sim[i, :])[::-1][1:11]).difference(np.argsort(W_new_sim[i, :])[::-1][1:11]))]

np.mean(diff)
sum([1 for d in diff if d > 0])/len(diff)

print([vocab[i] for i in np.argsort(W_sim[indices_of_vocab['politics'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_new_sim[indices_of_vocab['politics'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_sim[indices_of_vocab['trump'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_new_sim[indices_of_vocab['trump'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_sim[indices_of_vocab['u.s.'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_new_sim[indices_of_vocab['u.s.'], :])[::-1][1:11]])

stop_words = pickle.load(gzip.open('data/stopwords.pkl.gz', 'rb'))

data = pickle.load(gzip.open('cnn.pkl.gz', 'rb'))
X_test = [data[i][1].astype('int32') for i in range(3)]

categories = ['world', 'us', 'politics']
os = open('result/gan/test_original.txt', 'w')
for i in range(3):
    os.write('category \"%s\":\n' %categories[i])
    for j in range(50):
        ind = np.argsort(X_test[i][j, :])[::-1]
        words = [vocab[k] for k in ind]
        words = [v for v in words if v not in stop_words]
        os.write(' '.join(words[:20]) + '\n')

os.close()

data_new = pickle.load(gzip.open('result/gan/output.pkl.gz', 'rb'))
X_test_new = [data_new[[k for k in range(30000) if (k%150)//50 == i][:X_test[i].shape[0]], :] for i in range(3)]
idf = np.load('idf.npy')
idf = idf[-1,:]/idf[-1,:].max()
X_test = [normalize(x*idf) for x in X_test]
X_test_all = np.vstack(X_test + X_test_new)

reduced_data = TSNE(n_components=2).fit_transform(X_test_all)

group_11 = (reduced_data[:2111, :])[np.linspace(0, 2110, 100, dtype='int32'), :]
group_12 = (reduced_data[2111:3211, :])[np.linspace(0, 1099, 100, dtype='int32'), :]
group_13 = (reduced_data[3211:5420, :])[np.linspace(0, 2208, 100, dtype='int32'), :]
group_21 = (reduced_data[5420:7531, :])[np.linspace(0, 2110, 100, dtype='int32'), :]
group_22 = (reduced_data[7531:8631, :])[np.linspace(0, 1099, 100, dtype='int32'), :]
group_23 = (reduced_data[8631:, :])[np.linspace(0, 2208, 100, dtype='int32'), :]

plt.clf()
fig, ax = plt.subplots()
ax.scatter(group_21[:,0], group_21[:,1], color=(0,0,1)) 
ax.scatter(group_22[:,0], group_22[:,1], color=(0,0,1))
ax.scatter(group_23[:,0], group_23[:,1], color=(0,0,1))
ax.scatter(group_11[:,0], group_11[:,1], color=(1,0,0)) 
ax.scatter(group_12[:,0], group_12[:,1], color=(1,0,0)) 
ax.scatter(group_13[:,0], group_13[:,1], color=(1,0,0)) 
plt.savefig('tsne1.pdf')
