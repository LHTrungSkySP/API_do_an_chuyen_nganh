import pickle

# xxx = pickle.load(open(r'module\X_data_svd.pkl', 'rb'))
# print(len(xxx[0]))
X_data_tfidf_ngram_svd=pickle.load(open(r'module\X_data_svd.pkl', 'rb'))
y_data=pickle.load(open(r'module\y_data_n.pkl', 'rb'))
print(y_data)

X_test=pickle.load(open(r'module\X_test_svd.pkl', 'rb'))
y_test=pickle.load(open(r'module\y_test_n.pkl', 'rb'))

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
model=clf.fit(X_data_tfidf_ngram_svd, y_data)

pickle.dump(model, open(r'module\AI.pkl', 'wb'))

score = clf.score(X_data_tfidf_ngram_svd, y_data)
print('Accuracy of sklearn: {0}%'.format(score*100))