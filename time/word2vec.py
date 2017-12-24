import gensim, os, re, nltk, numpy as np, pickle, gzip
from gensim.models.word2vec import Word2Vec, LineSentence
from nltk.tokenize.moses import MosesDetokenizer

categories = ['Entertainment', 'Ideas', 'Politics', 'US', 'World']
for category in categories:
    w = open(('data/cleaned/%s.txt' %category), 'w')
    if os.path.exists('data/articles/%s' %category):
        files = os.listdir('data/articles/%s' %category)
        for file in files:
            for line in open(('data/articles/%s/%s' %(category, file)), 'r'):
                w.write(line)
    category = category.lower()
    if os.path.exists('data/articles/%s' %category):
        files = os.listdir('data/articles/%s' %category)
        for file in files:
            for line in open(('data/articles/%s/%s' %(category, file)), 'r'):
                w.write(line)
    w.close()

detokenizer = MosesDetokenizer()
stemmer = nltk.stem.porter.PorterStemmer()
stemming_dict = dict()

for category in categories:
    docs = open('data/cleaned/'+category+'.txt', 'r')
    # cleaned_docs = open('data/cleaned/cleaned_'+category+'.txt', 'w')
    for d in docs:
        doc = re.sub('e mail', 'e-mail', d)
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
            # cleaned_sent = detokenizer.detokenize([re.sub('^\'|\$', '', stemmer.stem(word)) for word in tokenized_sent], return_str=True)
            # cleaned_sent = re.sub(' n\'t', 'n\'t', cleaned_sent)
            # cleaned_docs.write(cleaned_sent + ' ')
        # cleaned_docs.write('\n')
    # cleaned_docs.close()

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

seeds = [1, 123, 888, 1234, 8888]
entertainment_model, ideas_model, world_model, us_model, politics_model, all_model = [], [], [], [], [], []
for i in range(5):
    entertainment_model += [Word2Vec(LineSentence('data/cleaned/cleaned_Entertainment_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]
    ideas_model += [Word2Vec(LineSentence('data/cleaned/cleaned_Ideas_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]
    world_model += [Word2Vec(LineSentence('data/cleaned/cleaned_World_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]
    us_model += [Word2Vec(LineSentence('data/cleaned/cleaned_US_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]
    politics_model += [Word2Vec(LineSentence('data/cleaned/cleaned_Politics_train_in_sent.txt'), size=300, window=5, min_count=5, workers=4)]
    # os.system('cat data/cleaned/cleaned_Entertainment_train_in_sent.txt data/cleaned/cleaned_Ideas_train_in_sent.txt data/cleaned/cleaned_World_train_in_sent.txt '
    #     + 'data/cleaned/cleaned_US_train_in_sent.txt data/cleaned/cleaned_Politics_train_in_sent.txt > data/cleaned/cleaned_all_train_in_sent.txt')
    all_model += [Word2Vec(LineSentence('data/cleaned/cleaned_all_train_in_sent.txt'), seed=seeds[i], size=300, window=5, min_count=5, workers=4)]

vocab = list(set(world_model[0].vocab.keys()).union(us_model[0].vocab.keys()).union(politics_model[0].vocab.keys()).union(all_model[0].vocab.keys())
    .union(entertainment_model[0].vocab.keys()).union(ideas_model[0].vocab.keys()))
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

DF = np.zeros((5, len(vocab)), dtype='float32')
N = [0]*5
for i in range(5):
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

idf = np.zeros((6, len(vocab)), dtype='float32')
for i in range(5):
    idf[i,:] = np.log(N[i]/np.maximum(1., DF[i,:]))

idf[5,:] = np.log(sum(N)/np.maximum(1., DF.sum(axis=0)))
np.save('idf.npy', idf)

X = []
for i in range(5):
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
    X += [[np.array(X_i_train).astype('int32'), np.array(X_i_test).astype('int32')]]

import pickle, gzip
f = gzip.open('time.pkl.gz', 'wb')
pickle.dump(X, f, protocol=2)
f.close()

categories += ['all']
models = [entertainment_model, ideas_model, politics_model, us_model, world_model, all_model]
vocab = [line.strip() for line in open('data/vocab.txt', 'r')]
for j in range(5):
    W = []
    for i in range(6):
        W_i = []
        model = models[i][j]
        for k in range(len(vocab)):
            W_i += [model[vocab[k]]] if vocab[k] in model else [np.zeros(300)]
        W += [np.array(W_i).astype('float32')]
    f = gzip.open('data/models/word2vec-%s.pkl.gz' %seeds[j], 'wb')
    pickle.dump(W, f, protocol=2)
    f.close()
