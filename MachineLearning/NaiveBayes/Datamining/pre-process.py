import nltk
import pickle
import re
import os
import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
from sklearn import preprocessing


def get_data(root = './data'):
    X_train = []
    y_train = []
    labels = os.listdir(root)
    for label in labels:
        list_file = os.listdir(os.path.join(root, label))
        for file in list_file:
            X = ""
            # print(file, label)
            with open(os.path.join(root, label, file), 'r', encoding="utf8", errors='ignore') as f_r:
                data = f_r.read().splitlines()
            
            for content in data:
                if content == '':
                    continue
                X += content + " "
            
            X_train.append(X)
            y_train.append(label)

    return X_train, y_train

def get_data_csv(root = './data_crawled.csv'):
    data = pd.read_csv(root)
    X_test = []
    y_test = []
    label = {1:'bussiness', 2 : 'entertainment', 3:'politics', 4:'sport', 5:'tech'}
    contents = data['content']
    category = data['category']
    for i in range(len(contents)):
        X_test.append(contents[i])
        y_test.append(label[int(category[i])])
    
    return X_test, y_test


class Pre_process:
    
    def __init__(self, X_train, y_train, X_test = None, y_test = None):
        self.X_train = X_train
        self.y_train = y_train
        if X_test is not None and y_test is not None:
            self.X_test = X_test
            self.y_test = y_test
        self.stopwords = set(stopwords.words('english'))
        # self.stopwords = set(open('./stopwords.txt', 'r').read().splitlines())

    def _lowercase(self, input_str):
        return input_str.lower()

    def _remove_numbers(self, input_str):
        return re.sub(r'\d+', '', input_str)
    
    def _remove_punctuation(self, input_str):
        return input_str.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_whitespaces(self, input_str):
        return input_str.strip()

    def _stemming(self, input_str):
        output_str = ''
        stemmer = PorterStemmer()
        input_str = word_tokenize(input_str)

        for word in input_str:
            output_str += stemmer.stem(word) + ' '
        
        return output_str 

    def _lemmatization(self, input_str):
        output_str = ''
        lemmatizer = WordNetLemmatizer()
        input_str = word_tokenize(input_str)

        for word in input_str:
            output_str += lemmatizer.lemmatize(word) + ' '
        return output_str
        
    def _remove_stopwords(self, input_str):
        output_str = ''
        input_str = word_tokenize(input_str)
        for word in input_str:
            if word not in self.stopwords:
                output_str += word + ' '
        return output_str
    
    def pre_process_data(self, train = True):
        if train == True:
            for i in tqdm(range(len(self.X_train))):
                self.X_train[i] = self._remove_punctuation(self.X_train[i])
                self.X_train[i] = self._lowercase(self.X_train[i])
                self.X_train[i] = self._remove_numbers(self.X_train[i])
                self.X_train[i] = self._remove_whitespaces(self.X_train[i])
                self.X_train[i] = self._stemming(self.X_train[i])
                self.X_train[i] = self._lemmatization(self.X_train[i])
                self.X_train[i] = self._remove_stopwords(self.X_train[i])
            pickle.dump(self.X_train, open('./X_train.pkl', 'wb'))
        else:
            for i in tqdm(range(len(self.X_test))):
                self.X_test[i] = self._remove_punctuation(self.X_test[i])
                self.X_test[i] = self._lowercase(self.X_test[i])
                self.X_test[i] = self._remove_numbers(self.X_test[i])
                self.X_test[i] = self._remove_whitespaces(self.X_test[i])
                self.X_test[i] = self._stemming(self.X_test[i])
                self.X_test[i] = self._lemmatization(self.X_test[i])
                self.X_test[i] = self._remove_stopwords(self.X_test[i])
            pickle.dump(self.X_test, open('./X_test.pkl', 'wb'))

    def _label_encoder(self):
        encoder = preprocessing.LabelEncoder()
        self.y_train = encoder.fit_transform(self.y_train)  
        self.y_test = encoder.fit_transform(self.y_test)
        print(encoder.classes_)
        pickle.dump(self.y_train, open('./y_train.pkl', 'wb'))
        pickle.dump(self.y_test, open('./y_test.pkl', 'wb'))

    def run(self):
        print('Pre-processing train data ...')
        self.pre_process_data()
        print('Pre-processing test data ...')
        self.pre_process_data(train=False)
        self._label_encoder()


class Model:
    
    def __init__(self):
        self.X_train = pickle.load(open('./X_test.pkl', 'rb'))
        self.y_train = pickle.load(open('./y_test.pkl', 'rb'))
        self.X_test = pickle.load(open('./X_train.pkl', 'rb'))
        self.y_test = pickle.load(open('./y_train.pkl','rb'))
    
    def _tfidf(self):
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(1, 3))
        tfidf_vect_ngram.fit(self.X_train)
        self.X_train = tfidf_vect_ngram.transform(self.X_train)
        self.X_test = tfidf_vect_ngram.transform(self.X_test)

    def _truncated_svd():
        return
    
    def _train_model(self, model = MultinomialNB(), test_size = 0.2):
        # X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size = test_size, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions_test = model.predict(self.X_test)

        print('===================================')
        print("results validation")
        print('Accuracy score:  ', accuracy_score(self.y_test, predictions_test))
        print('Precision score: ', precision_score(self.y_test, predictions_test, average = None))
        print('Recall score:    ', recall_score(self.y_test, predictions_test, average = None))
        print('===================================')

        
    def run_NB(self):
        print("Word embedding ...")
        self._tfidf()
        print('Training ...')
        self._train_model()

    def run_DT(self):
        print("Training decition tree ...")
        self._train_model()


if __name__ == '__main__':
    X_train, y_train = get_data()
    X_test, y_test = get_data_csv()
    pre_process = Pre_process(X_train, y_train, X_test = X_test, y_test = y_test)
    pre_process.run()
    model = model.Model()
    model.run_NB()
    
