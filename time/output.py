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

print([vocab[i] for i in np.argsort(W_sim[indices_of_vocab['obama'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_new_sim[indices_of_vocab['obama'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_sim[indices_of_vocab['trump'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_new_sim[indices_of_vocab['trump'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_sim[indices_of_vocab['u.s.'], :])[::-1][1:11]])
print([vocab[i] for i in np.argsort(W_new_sim[indices_of_vocab['u.s.'], :])[::-1][1:11]])

stop_words = pickle.load(gzip.open('data/stopwords.pkl.gz', 'rb'))

data = pickle.load(gzip.open('time.pkl.gz', 'rb'))
X_test = [data[i][1].astype('int32') for i in range(5)]

categories = ['Entertainment', 'Ideas', 'Politics', 'US', 'World']
os = open('result/gan/test_original.txt', 'w')
for i in range(5):
    os.write('category \"%s\":\n' %categories[i])
    for j in range(50):
        ind = np.argsort(X_test[i][j, :])[::-1]
        words = [vocab[k] for k in ind]
        words = [v for v in words if v not in stop_words]
        os.write(' '.join(words[:20]) + '\n')

os.close()

data_new = pickle.load(gzip.open('result/gan/output.pkl.gz', 'rb'))
X_test_new = [data_new[[k for k in range(20000) if (k%250)//50 == i][:X_test[i].shape[0]], :] for i in range(5)]
idf = np.load('idf.npy')
idf = idf[-1,:]/idf[-1,:].max()
X_test = [normalize(x*idf) for x in X_test]
X_test_all = np.vstack(X_test + X_test_new)

reduced_data = TSNE(n_components=2).fit_transform(X_test_all)
n_test = [x.shape[0] for x in X_test]*2

group_1 = np.vstack([(reduced_data[sum(n_test[:i]):sum(n_test[:(i+1)]), :])[np.linspace(0, n_test[i]-1, 100, dtype='int32'), :] for i in range(5)])
group_2 = np.vstack([(reduced_data[sum(n_test[:i]):sum(n_test[:(i+1)]), :])[np.linspace(0, n_test[i]-1, 100, dtype='int32'), :] for i in range(5, 10)])

plt.clf()
fig, ax = plt.subplots()
ax.scatter(group_2[:,0], group_2[:,1], color=(0,0,1)) 
ax.scatter(group_1[:,0], group_1[:,1], color=(1,0,0)) 
plt.savefig('tsne2.pdf')
