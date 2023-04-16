from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import time
import pickle
start_time = time.time()
X_data = pickle.load(open(r'module\trainAI\X_data.pkl', 'rb'))
y_data = pickle.load(open(r'module\trainAI\y_data.pkl', 'rb'))

X_test = pickle.load(open(r'module\trainAI\X_test.pkl', 'rb'))
y_test = pickle.load(open(r'module\trainAI\y_test.pkl', 'rb'))

# cho vào bảng
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram_fit=tfidf_vect_ngram.fit(X_data)
pickle.dump(tfidf_vect_ngram_fit, open(r'module\tfidf_vect_ngram_fit.pkl', 'wb'))

X_data_tfidf_ngram =  tfidf_vect_ngram.transform(X_data)
X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

# rút gọn
svd_ngram = TruncatedSVD(n_components=3000, random_state=42)
svd_ngram_fit=svd_ngram.fit(X_data_tfidf_ngram)
pickle.dump(svd_ngram_fit, open(r'module\svd_ngram_fit.pkl', 'wb'))

X_data_tfidf_ngram_svd = svd_ngram.transform(X_data_tfidf_ngram)
X_test_tfidf_ngram_svd = svd_ngram.transform(X_test_tfidf_ngram)

# chuyển label về số
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)

pickle.dump(X_data_tfidf_ngram_svd, open(r'module\X_data_svd.pkl', 'wb'))
pickle.dump(X_test_tfidf_ngram_svd, open(r'module\X_test_svd.pkl', 'wb'))
pickle.dump(y_data_n, open(r'module\y_data_n.pkl', 'wb'))
pickle.dump(y_test_n, open(r'module\y_test_n.pkl', 'wb'))

