import pickle
import time

import gensim  # thư viện NLP
import numpy as np
from pyvi import ViPosTagger, ViTokenizer  # thư viện NLP tiếng Việt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# xử dụng stop word loại bỏ từ không có ý nghĩa trong việc phân loại

def remove_stopwords(line):
    f = open("classify/stopword.txt",encoding = 'utf-8')
    stopword=f.read()
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    f.close()
    return ' '.join(words)

def processing_data(data):
    data = gensim.utils.simple_preprocess(data)
    data = ' '.join(data)
    data = ViTokenizer.tokenize(data)

    data=remove_stopwords(data)
    return data

def classify(str):
    xxx=[]
    xxx.append(processing_data(str))
    print("str : ",xxx)

    tfidf_vect_ngram = pickle.load(open(r'classify/tfidf_vect_ngram_fit.pkl', 'rb'))
    xxx_tfidf_ngram =  tfidf_vect_ngram.transform(xxx)
    print(xxx_tfidf_ngram)

    svd_ngram = pickle.load(open(r'classify/svd_ngram_fit.pkl', 'rb'))

    xxx_tfidf_ngram_svd = svd_ngram.transform(xxx_tfidf_ngram)
    print(xxx_tfidf_ngram_svd)
    model = pickle.load(open(r'classify/AI.pkl', 'rb'))

    score = model.predict_proba(xxx_tfidf_ngram_svd)
    score1 = model.predict(xxx_tfidf_ngram_svd)
    return score,score1

