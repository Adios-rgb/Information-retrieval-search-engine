import os
import re
import time
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

global processed_text_docs, original_text
processed_text_docs, original_text = [], []


def preprocess_text(text):
    if text != '':
        text = text.lower().strip()
        text = re.sub('\n',' ',text)
        text = re.sub("(\\d|\\W)+"," ", text).strip()
        text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        sent = ' '.join(text)
        return sent


def create_tfidf_features(corpus, max_features=1000, max_df=0.95, min_df=2):
    tfidf_vectorizor = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word',
                                       stop_words='english', ngram_range=(1, 1), max_features=max_features,
                                       norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                                       max_df=max_df, min_df=min_df)
    X = tfidf_vectorizor.fit_transform(corpus)
    return X, tfidf_vectorizor


def calculate_similarity(X, vectorizor, query, top_k=5):
    query_vec = vectorizor.transform(query)
    cosine_similarities = cosine_similarity(X,query_vec).flatten()
    most_similar_doc_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
    return (most_similar_doc_indices, cosine_similarities)


def show_similar_documents(df, cosine_similarities, similar_doc_indices):
    output = []
    counter = 1
    for index in similar_doc_indices:
        output.append(('Document-{}'.format(counter), df[index][: 2000]))
        counter += 1
    return output


def make_text_documents():
    original_text = []
    sentences = ''
    for idx, txt in enumerate(brown.sents()):
        sentences += ' '.join(txt)
        if (idx + 1) % 2000 == 0:
            original_text.append(sentences)
            sentences = ''
    processed_text_docs = []
    for doc_text in original_text:
        sent = preprocess_text(doc_text)
        processed_text_docs.append(sent)
    return processed_text_docs, original_text


def add_doc_to_original(doc):
    original_text.append(doc)


def add_doc_to_processed(text_processed):
    processed_text_docs.append(text_processed)


def get_similar_documents(query):
    processed_text_docs, original_text = make_text_documents()
    X_data, v = create_tfidf_features(processed_text_docs)
    features = v.get_feature_names()
    user_question = [query]
    search_start = time.time()
    sim_vects, cosine_similarities_ = calculate_similarity(X_data, v, user_question)
    search_time = time.time() - search_start
    result = show_similar_documents(original_text, cosine_similarities_, sim_vects)
    return result