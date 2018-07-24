
# coding: utf-8

# # Text Mining
# 
# ### Mohammad Sadegh Ghasemi
# #### this project is based on the "Text Analytics with Python" book
# #### Created for "Computational Data Mining", Kharazmi University, 96-1 

# In[57]:


# Importin required libraries/codes
import nltk
import re
import string

import numpy as np
from numpy.linalg import norm

import scipy.sparse as sp

import pandas as pd
from pprint import pprint

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# query section required functions
from normalization import parse_document, normalize_corpus, normalize_document
from feature_extractors import build_feature_matrix

# Topic Modeling
from gensim import corpora, models
from utils import low_rank_svd
from sklearn.decomposition import NMF


# ## Feature extraction functions

# In[2]:


# Feature Extraction codes

def bow_extractor(corpus, ngram_range=(1, 1)):

    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def tfidf_transformer(bow_matrix):

    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


def tfidf_extractor(corpus, ngram_range=(1, 1)):

    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def average_word_vectors(words, model, vocabulary, num_features):

    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)


def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model, num_features):

    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)]
                   if tfidf_vocabulary.get(word)
                   else 0 for word in words]
    word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}

    feature_vector = np.zeros((num_features,), dtype="float64")
    vocabulary = set(model.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary:
            word_vector = model[word]
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector, wts)

    return feature_vector


def tfidf_weighted_averaged_word_vectorizer(corpus, tfidf_vectors,
                                            tfidf_vocabulary, model, num_features):

    docs_tfidfs = [(doc, doc_tfidf)
                   for doc, doc_tfidf
                   in zip(corpus, tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
                                           model, num_features)
                for tokenized_sentence, tfidf in docs_tfidfs]
    return np.array(features)


def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
    print(df)


# ## Text Normalization functions

# In[3]:


def tokenize_text(text):
    sentence_tokens = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentence_tokens]
    return sentence_tokens, word_tokens


def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = list(filter(None, [pattern.sub('', token) for token in tokens]))
    return filtered_tokens


def remove_characters_before_tokenization(sentence, keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'  # add other characters here to remove them
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    else:
        PATTERN = r'[^a-zA-Z0-9 ]'  # only extract alpha-numeric characters
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence


def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


# ## loading a corpus 

# In[4]:


def get_corpus(name):

    corpus = []
    with open(name, 'r') as corpus_file:
        for doc in corpus_file:
            if doc.strip():
                corpus.append(parse_document(doc))
        corpus_file.close()
        return corpus


# ## searching between corpus documents 

# In[5]:


def search(corpus):
    """Search Demo function"""

    normalized_corpus = normalize_corpus(corpus)
    vectorizer, feature_matrix = build_feature_matrix(
        normalized_corpus, 'tfidf')

    q = input('Enter search query. Press "Enter" to stop: \n')

    while q != '':
        q = normalize_document(q)
        q_tfidf = vectorizer.transform([q])
        ans_mat = q_tfidf.dot(feature_matrix.transpose())
        ans_list = []
        for j in range(ans_mat.shape[1]):
            if ans_mat[0, j] > 0:
                ans_list.append((j, ans_mat[0, j]))
        ans_list.sort(key=lambda x: x[1], reverse=True)

        print()
        print('************ {} ************'.format(q))

        for item in ans_list[:5]:
            print()
            print('Document no. {}, rank: {}'.format(item[0], item[1]))
            print(corpus[item[0]][:])
            print()
            print()

        q = input('Enter search query. Press "Enter" to stop: \n')
    print()


# ## Topic modeling functions

# In[48]:


def print_topics_gensim(topic_model, total_topics=1,
                        weight_threshold=0.0001,
                        display_weights=False,
                        num_terms=None):

    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic = [(word, round(wt, 2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]
        if display_weights:
            print('Topic #' + str(index + 1) + ' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #' + str(index + 1) + ' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)
        print()


def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1])
                               for row
                               in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index])
                               for wt, index
                               in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row])
                             for row
                             in sorted_indices])

    topics = [np.vstack((terms.T,
                         term_weights.T)).T
              for terms, term_weights
              in zip(sorted_terms, sorted_weights)]

    return topics


def print_topics_udf(topics, total_topics=1,
                     weight_threshold=0.0001,
                     display_weights=False,
                     num_terms=None):

    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        topic = [(word, round(wt, 2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        if display_weights:
            print('Topic #' + str(index + 1) + ' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #' + str(index + 1) + ' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)
        print()


# # Usage examples

# ## Tokenization

# In[6]:


# Word TOKENIZATION
sentence = "The brown fox wasn't that quick and he couldn't win the race"
words = nltk.word_tokenize(sentence)
print(words)


# In[7]:


# Sentence TOKENIZATION
sample_text = 'We will discuss briefly about the basic syntax, structure and design philosophies.  There is a defined hierarchical syntax for Python code which you should remember  when writing code! Python is a really powerful programming language!'
sentences = nltk.sent_tokenize(sample_text)
print(sample_text)
print('\nTotal sentences in sample_text:', len(sentences))
print()
pprint(sentences[0])


# In[8]:


# Sentence and Word TOKENIZATION
another_text = 'We _ ##will discuss @*briefly about the basic syntax,\
 structure and design philosophies. \
 There is a defined $hierarchical syntax for Python code which you should remember \
 when writing code! @@Python (is) a **really** powerful programming language!@@'
sentences, words = tokenize_text(another_text)
print(sentences[0])
print()
print(words[0])


# ## Normalization

# ### Cleaning

# In[9]:


# Cleaning
cleaned_sentences = [remove_characters_before_tokenization(s) for s in sentences]
print(sentences[2])
print()
print(cleaned_sentences[2])


# ### Removing StopWords

# In[10]:


# Removing Stopwords

words = ['am', 'a', 'then', 'so', 'available', 'google']
print(remove_stopwords(words))


# ### Removing repeated characters

# In[11]:


# removing repeated characters

sample_sentence = 'Myyyy nameee is realllllyyy amaaazingggg'
sample_sentence_tokens = tokenize_text(sample_sentence)[0]

print(remove_repeated_characters(sample_sentence_tokens))


# ### Stemming

# In[12]:


# porter stemmer

ps = PorterStemmer()

print(
    ps.stem('activation'),
    ps.stem('active'),
    ps.stem('actively'),
    ps.stem('activities'))

print(
    ps.stem('jumping'),
    ps.stem('jumps'),
    ps.stem('jumped'))

print(ps.stem('lying'))

print(ps.stem('strange'))


# ### Lemmatization

# In[13]:


# lemmatization

wnl = WordNetLemmatizer()


# In[14]:


# lemmatize nouns
print(wnl.lemmatize('cars', 'n'))
print(wnl.lemmatize('men', 'n'))


# In[15]:


# lemmatize verbs
print(wnl.lemmatize('running', 'v'))
print(wnl.lemmatize('ate', 'v'))


# In[16]:


# lemmatize adjectives
print(wnl.lemmatize('saddest', 'a'))
print(wnl.lemmatize('fancier', 'a'))


# In[17]:


# ineffective lemmatization
print(wnl.lemmatize('ate', 'n'))
print(wnl.lemmatize('fancier', 'v'))


# ## Feature extraction

# ### Vector Space Model

# In[18]:


# Vector Space Model
CORPUS = [
    'the sky is blue',
    'sky is blue and sky is beautiful',
    'the beautiful sky is so blue',
    'i love blue cheese'
]

bow_vectorizer, bow_features = bow_extractor(CORPUS)

feature_names = bow_vectorizer.get_feature_names()
features = bow_features.todense()

display_features(features, feature_names)


# In[19]:


new_doc = ['loving this blue sky today']
new_doc_features = bow_vectorizer.transform(new_doc)
new_doc_features = new_doc_features.todense()

display_features(new_doc_features, feature_names)


# ## Term Frequency Inverse Document Frequency

# In[20]:


# TF-IDF
feature_names = bow_vectorizer.get_feature_names()

tfidf_trans, tdidf_features = tfidf_transformer(bow_features)
features = np.round(tdidf_features.todense(), 2)
display_features(features, feature_names)


# In[21]:


# New document
# new_doc = ['loving this blue sky today']
nd_tfidf = tfidf_trans.transform(new_doc_features)
nd_features = np.round(nd_tfidf.todense(), 2)
display_features(nd_features, feature_names)


# # TF-IDF (Custom computation)

# In[22]:


# Computing TF-IDF Matrices

feature_names = bow_vectorizer.get_feature_names()

# compute term frequency
tf = bow_features.todense()
tf = np.array(tf, dtype='float64')

# show term frequencies
display_features(tf, feature_names)


# In[23]:


# build the document frequency matrix
df = np.diff(sp.csc_matrix(bow_features, copy=True).indptr)
df = 1 + df  # to smoothen idf later

# show document frequencies
display_features([df], feature_names)


# In[24]:


# compute inverse document frequencies
total_docs = 1 + len(CORPUS)
idf = 1.0 + np.log(float(total_docs) / df)

# show inverse document frequencies
display_features([np.round(idf, 2)], feature_names)


# In[25]:


# compute idf diagonal matrix
total_features = bow_features.shape[1]
idf_diag = sp.spdiags(idf, diags=0, m=total_features, n=total_features)
idf = idf_diag.todense()

# print(the idf diagonal matrix)
print(np.round(idf, 2))


# In[26]:


# compute tfidf feature matrix
tfidf = tf * idf

# show tfidf feature matrix
display_features(np.round(tfidf, 2), feature_names)


# In[27]:


# compute L2 norms
norms = norm(tfidf, axis=1)

# print(norms for each document)
print(np.round(norms, 2))


# In[28]:


# compute normalized tfidf
norm_tfidf = tfidf / norms[:, None]

# show final tfidf feature matrix
display_features(np.round(norm_tfidf, 2), feature_names)


# In[29]:


# compute new doc term freqs from bow freqs
nd_tf = new_doc_features
nd_tf = np.array(nd_tf, dtype='float64')

# compute tfidf using idf matrix from train corpus
nd_tfidf = nd_tf * idf
nd_norms = norm(nd_tfidf, axis=1)
norm_nd_tfidf = nd_tfidf / nd_norms[:, None]

# show new_doc tfidf feature vector
display_features(np.round(norm_nd_tfidf, 2), feature_names)


# ## Text Query

# In[30]:


corpus = get_corpus('corpus400.txt')
search(corpus)


# ## Topic Modeling

# In[33]:


toy_corpus = ["The fox jumps over the dog",
              "The fox is very clever and quick",
              "The dog is slow and lazy",
              "The cat is smarter than the fox and the dog",
              "Python is an excellent programming language",
              "Java and Ruby are other programming languages",
              "Python and Java are very popular programming languages",
              "Python programs are smaller than Java programs"]


# ### LSI topic model

# In[56]:


norm_tokenized_corpus = normalize_corpus(toy_corpus, tokenize=True)
norm_tokenized_corpus


# In[39]:


dictionary = corpora.Dictionary(norm_tokenized_corpus)
print(dictionary.token2id)


# In[42]:


corpus = [dictionary.doc2bow(text) for text in norm_tokenized_corpus]
corpus


# In[43]:


tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

total_topics = 2

lsi = models.LsiModel(corpus_tfidf,
                      id2word=dictionary,
                      num_topics=total_topics)

for index, topic in lsi.print_topics(total_topics):
    print('Topic #' + str(index + 1))
    print(topic)
    print()


# In[46]:


print_topics_gensim(topic_model=lsi,
                    total_topics=total_topics,
#                     num_terms=5,
                    display_weights=True)


# ### LSI custom built topic model (using SVD)

# In[51]:


norm_corpus = normalize_corpus(toy_corpus)

vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus,
                                                feature_type='tfidf')
td_matrix = tfidf_matrix.transpose()

td_matrix = td_matrix.multiply(td_matrix > 0)

total_topics = 2
feature_names = vectorizer.get_feature_names()

u, s, vt = low_rank_svd(td_matrix, singular_count=total_topics)
weights = u.transpose() * s[:, None]

topics = get_topics_terms_weights(weights, feature_names)

print_topics_udf(topics=topics,
                 total_topics=total_topics,
                 weight_threshold=0.15,
                 display_weights=True)


# ### NMF

# In[55]:


norm_corpus = normalize_corpus(toy_corpus)
vectorizer, tfidf_matrix = build_feature_matrix(norm_corpus,
                                                feature_type='tfidf')
total_topics = 2
nmf = NMF(n_components=total_topics,
          random_state=42, alpha=.1, l1_ratio=.5)
nmf.fit(tfidf_matrix)

feature_names = vectorizer.get_feature_names()
weights = nmf.components_

topics = get_topics_terms_weights(weights, feature_names)
print_topics_udf(topics=topics,
                 total_topics=total_topics,
                 num_terms=None,
                 display_weights=True)

