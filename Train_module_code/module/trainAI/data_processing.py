from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP

import os 
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))

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
    data = ' '.join(data)
    data = gensim.utils.simple_preprocess(data)
    data = ' '.join(data)
    data = ViTokenizer.tokenize(data)
    data=remove_stopwords(data)
    return data

# tiền xử lý ( loại bỏ ký tự đặc biệt, số, nối các từ đơn thành từ ghép có nghĩa)
def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        # liệt kê các file nằm trong folder
        file_paths = os.listdir(os.path.join(folder_path, path))
        # print(file_paths)
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines=processing_data(lines)
                X.append(lines)
                y.append(path)
    return X, y



# đường dẫn data train
import pickle

# xử lý dữ liệu train
train_path=r'assets\Train_Full'
X_data, y_data = get_data(train_path)

pickle.dump(X_data, open(r'module\trainAI\X_data.pkl', 'wb'))
pickle.dump(y_data, open(r'module\trainAI\y_data.pkl', 'wb'))

# xử lý dữ liệu test
test_path=r'assets\Test_Full'
X_test,y_test= get_data(test_path)

pickle.dump(X_test, open(r'module\trainAI\X_test.pkl', 'wb'))
pickle.dump(y_test, open(r'module\trainAI\y_test.pkl', 'wb'))


