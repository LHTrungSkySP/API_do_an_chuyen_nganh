from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import time
import pickle
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
# xử dụng stop word loại bỏ từ không có ý nghĩa trong việc phân loại
f = open("assets\Stopword\stopword.txt",encoding = 'utf-8')
stopword=f.read()
def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)
def processing_data(data):
    data = gensim.utils.simple_preprocess(data)
    data = ' '.join(data)
    data = ViTokenizer.tokenize(data)

    data=remove_stopwords(data)
    return data

xxx=[]
str='Thủ tướng Đức nhận lời tham dự lễ kỷ niệm D-Day Thủ tướng Gerhard Schroeder sẽ trở thành nguyên thủ Đức đầu tiên tham dự lễ kỷ niệm ngày quân đồng minh đổ bộ lên bãi biển Normandy trong Thế chiến II (mang mật danh D-Day) vào tháng 6 tới. Ông đã chấp nhận lời mời tham gia lễ kỷ niệm 60 năm ngày D-Day của Tổng thống Pháp Jacque Chirac. Phát ngôn viên của Berlin cho biết: "Tổng thống Chirac đã mời Thủ tướng Schroeder từ trước lễ Giáng sinh và ông đã nhận lời ngay. Thủ tướng cảm thấy rất vui khi được mời". Năm 1994, cố tổng thống Pháp Francois Mitterrand đã không mời cựu thủ tướng Đức Helmut Kohl đến dự lễ kỷ niệm 50 năm sự kiện D-Day. Giới chức Pháp tuyên bố rằng, việc mời thủ tướng Schroeder tham dự lễ kỷ niệm 60 năm sự kiện D-Day là một hành động mang tính biểu tượng nhằm củng cố bầu không khí hòa bình lâu dài giữa hai nước.'
xxx.append(processing_data(str))

tfidf_vect_ngram = pickle.load(open(r'module\tfidf_vect_ngram_fit.pkl', 'rb'))
xxx_tfidf_ngram =  tfidf_vect_ngram.transform(xxx)
print(xxx_tfidf_ngram)

svd_ngram = pickle.load(open(r'module\svd_ngram_fit.pkl', 'rb'))

xxx_tfidf_ngram_svd = svd_ngram.transform(xxx_tfidf_ngram)

from sklearn.naive_bayes import BernoulliNB

model = pickle.load(open(r'module\AI.pkl', 'rb'))

score = model.predict_proba(xxx_tfidf_ngram_svd)
score1 = model.predict(xxx_tfidf_ngram_svd)
print(score)
print(score1)


