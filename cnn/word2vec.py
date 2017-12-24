# do this first and zip the text files

import gensim, os, re, nltk, numpy as np, pickle, gzip, string
from gensim.models.word2vec import Word2Vec, LineSentence
from nltk.tokenize.moses import MosesDetokenizer

world = open('data/cleaned/world.txt', 'w')
us = open('data/cleaned/us.txt', 'w')
politics = open('data/cleaned/politics.txt', 'w')

for r in os.walk('/home/bwang/gan/cnn/articles/'):
    category = re.sub('/home/bwang/gan/cnn/articles/', '', r[0])
    docs = sum([open(r[0]+'/'+f, 'r').readlines() for f in r[2]], [])
    if category.startswith('world') or category in set(['asia', 'europe', 'middleeast', 'africa', 'americas']):
        for doc in docs:
            world.write(doc)
    elif category == 'us':
        for doc in docs:
            us.write(doc)
    elif category == 'politics':
        for doc in docs:
            politics.write(doc)

world.close()
us.close()
politics.close()

categories = ['world', 'us', 'politics']
detokenizer = MosesDetokenizer()
stemmer = nltk.stem.porter.PorterStemmer()
stemming_dict = dict()

for category in categories:
    docs = open('data/cleaned/'+category+'.txt', 'r')
    cleaned_docs = open('data/cleaned/cleaned_'+category+'.txt', 'w')
    for d in docs:
        doc = re.sub('cnnpolitics\.com', '', d)
        doc = re.sub('e mail', 'e-mail', doc)
        doc = re.sub('\'m|\'re|\'s|\'ll|\'d|\'ve', '', doc)
        sentences = nltk.sent_tokenize(doc)
        if len(sentences) <= 5:
            continue
        for sent in sentences:
            tokenized_sent = nltk.word_tokenize(sent)
            for word in tokenized_sent:
                stemmed_word = stemmer.stem(word)
                if stemmed_word not in stemming_dict:
                    stemming_dict[stemmed_word] = {word : 1}
                elif word not in stemming_dict[stemmed_word]:
                    stemming_dict[stemmed_word][word] = 1
                else:
                    stemming_dict[stemmed_word][word] += 1
            cleaned_sent = detokenizer.detokenize([re.sub('^\'|\$', '', stemmer.stem(word)) for word in tokenized_sent], return_str=True)
            cleaned_sent = re.sub(' n\'t', 'n\'t', cleaned_sent)
            cleaned_docs.write(cleaned_sent + ' ')
        cleaned_docs.write('\n')
    cleaned_docs.close()

def key_for_max_val(d):
    val = -np.inf
    kk = None
    for k in d:
        if d[k] > val:
            kk = k
            val = d[k]
    return kk

stemming_dict = {s : key_for_max_val(stemming_dict[s]) for s in stemming_dict}
f = gzip.open('data/stemming_dict.pkl.gz', 'wb')
pickle.dump(stemming_dict, f, protocol=2)
f.close()

for category in categories:
    lines = open('data/cleaned/cleaned_'+category+'.txt', 'r').readlines()
    lines = list(set(lines))
    train_indices = np.random.choice(len(lines), int(0.8*len(lines)), replace=False)
    cleaned_train = open('data/cleaned/cleaned_'+category+'_train.txt', 'w')
    cleaned_test = open('data/cleaned/cleaned_'+category+'_test.txt', 'w')
    for i in range(len(lines)):
        if i in train_indices:
            cleaned_train.write(lines[i].strip() + '\n')
        else:
            cleaned_test.write(lines[i].strip() + '\n')
    cleaned_train.close()
    cleaned_test.close()

for category in categories:
    lines = open('data/cleaned/cleaned_'+category+'_train.txt', 'r').readlines()
    cleaned_docs_in_sent = open('data/cleaned/cleaned_'+category+'_train_in_sent.txt', 'w')
    for line in lines:
        sentences = nltk.sent_tokenize(line)
        for s in sentences:
            cleaned_docs_in_sent.write(re.sub('\.+$', '', s.strip()) + '\n')
    cleaned_docs_in_sent.close()

os.system('cat data/cleaned/cleaned_world_train_in_sent.txt data/cleaned/cleaned_us_train_in_sent.txt data/cleaned/cleaned_politics_train_in_sent.txt > ' + 
    'data/cleaned/cleaned_all_train_in_sent.txt')

seeds = [1, 123, 888, 1234, 8888]
world_model, us_model, politics_model, all_model = [], [], [], []
for i in range(5):
    world_model += [Word2Vec(LineSentence('data/cleaned/cleaned_world_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]
    us_model += [Word2Vec(LineSentence('data/cleaned/cleaned_us_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]
    politics_model += [Word2Vec(LineSentence('data/cleaned/cleaned_politics_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]
    all_model += [Word2Vec(LineSentence('data/cleaned/cleaned_all_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]

vocab = list(set(world_model[0].vocab.keys()).intersection(us_model[0].vocab.keys()).intersection(politics_model[0].vocab.keys()).intersection(all_model[0].vocab.keys()))
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

DF = np.zeros((3, len(vocab)), dtype='float32')
N = [0]*3
for i in range(3):
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

idf = np.zeros((4, len(vocab)), dtype='float32')
for i in range(3):
    idf[i,:] = np.log(N[i]/np.maximum(1., DF[i,:]))

idf[3,:] = np.log(sum(N)/np.maximum(1., DF.sum(axis=0)))
np.save('idf.npy', idf)

X = []
for i in range(3):
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
        X_i_train += [tf_per_doc]
    for line in cleaned_test:
        tf_per_doc = [0]*len(vocab)
        sentences = nltk.sent_tokenize(line.strip())
        for sent in sentences:
            for w in (re.sub('\.+$', '', sent)).split(' '):
                if w in indices_of_vocab:
                    tf_per_doc[indices_of_vocab[w]] += 1
        X_i_test += [tf_per_doc]
    X += [[np.array(X_i_train).astype('uint16'), np.array(X_i_test).astype('uint16')]]

f = gzip.open('cnn.pkl.gz', 'wb')
pickle.dump(X, f, protocol=2)
f.close()

models = [world_model, us_model, politics_model, all_model]
vocab = [line.strip() for line in open('data/vocab.txt', 'r')]
for j in range(5):
    W = []
    for i in range(4):
        W_i = []
        model = models[i][j]
        for k in range(len(vocab)):
            W_i += [model[vocab[k]]] if vocab[k] in model else [np.zeros(300)]
        W += [np.array(W_i).astype('float32')]
    f = gzip.open('data/models/word2vec-%s.pkl.gz' %seeds[j], 'wb')
    pickle.dump(W, f, protocol=2)
    f.close()

import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')).union(list(string.ascii_lowercase)).union([l + '.' for l in string.ascii_lowercase])
f = gzip.open('data/stopwords.pkl.gz', 'wb')
pickle.dump(stop_words, f, protocol=2)
f.close()
