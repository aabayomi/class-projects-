from __future__ import division  
import csv
import  os
import nltk 
import pandas as pd
import numpy as np 
from sklearn.utils import shuffle
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict, Counter
from nltk.tokenize import TweetTokenizer
from nltk.tag import StanfordPOSTagger
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]

def train(message,label):
    NaiveBayesClassifier = MultinomialNB().fit(message,label)
    LogisticRegressionClassifier = LogisticRegression().fit(message,label)
    return NaiveBayesClassifier,LogisticRegressionClassifier
     
java_path = "C:\Program Files\Java\jdk1.8.0_111\\bin\java.exe"
os.environ['JAVAHOME'] = java_path
st = StanfordPOSTagger('C:/Users/Abayomi/Desktop/machine learning/stanford-postagger-full-2015-12-09/models/english-bidirectional-distsim.tagger ','C:/Users/Abayomi/Desktop/machine learning/stanford-postagger-full-2015-12-09\
/stanford-postagger.jar') 

#data_file = "C:/Users/Abayomi/Desktop/machine learning/Resources-for-Projects-of-Machine-Learning-master/Datasets/sentiment/dataset1/Datasets.txt"#'data/labeledTrainData.tsv'
NRCLexicon = "C:/Users/Abayomi/Desktop/machine learning/NRC/NRCemotionlexiconword.txt"
#data_file= "C:/Users/Abayomi/Desktop/machine learning/Resources-for-Projects-of-Machine-Learning-master/Datasets/sentiment/dataset2/amazon_cells_labelled.txt"
data_file= "C:/Users/Abayomi/Desktop/machine learning/Resources-for-Projects-of-Machine-Learning-master/Datasets/sentiment/dataset2/imdb_labelled.txt"
#data_file= "C:/Users/Abayomi/Desktop/machine learning/Resources-for-Projects-of-Machine-Learning-master/Datasets/sentiment/dataset2/yelp_labelled.txt"

datafile = pd.read_csv(data_file, header = 0, delimiter= "\t", quoting = 3, names=['label','text']    )

datafile = shuffle(datafile) # radomising the datasets 
tweets = datafile['label'] #.head()
st.tag(tweets)
y = []
for i in datafile['text']:
    y.append(i)
y = np.array(y)

wordList = defaultdict(list)     # world list of tagged speech 
emotionList = defaultdict(list)
with open('C:/Users/Abayomi/Desktop/machine learning/NRC/NRCemotionlexiconword.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    headerRows = [i for i in range(0, 46)]
    for row in headerRows:
        next(reader)
    for word, emotion, present in reader:
        if int(present) == 1:
            wordList[word].append(emotion)
            emotionList[emotion].append(word)
            
token_t = TweetTokenizer()                #word_tokenize()
def generate_emotion_count(string, tokenizer):
    emoCount = Counter()
    for token in token_t.tokenize(string):
        token = token.lower()
        emoCount += Counter(wordList[token])
    return emoCount, token
emotionCounts = [generate_emotion_count(tweet, token_t) for tweet in tweets]
emotion_df = pd.DataFrame(emotionCounts, index=tweets.index)
emotion_df = emotion_df.fillna(0)

t = []
f = []
c = []
k = []
for i in emotionCounts:
    for j in i:
        t.append("-".join(j))
    f.extend(t)
    c.append(f)
    b = []
    a = []
h = []
for i in c:
    h.append(" ".join(i))
samples_proportion = 0.8    # spliting the data sets 
train_size = int(len(h) * samples_proportion)
train_set, test_set = h[:train_size], h[train_size:]
train_label, test_label = y[:train_size], y[train_size:]
bow_transformer = CountVectorizer(analyzer=preprocess).fit(train_set) #Preprocess Training Set
messages_bow = bow_transformer.transform(train_set)


message_test = CountVectorizer(analyzer=preprocess).fit(test_set)
bow_test = message_test.transform(test_set)

NaiveBayesClassifier = train(messages_bow, train_label)           # naive bayes classifier 
Naive_predictions = NaiveBayesClassifier.predict(bow_test)

LogisticRegressionClassifier= train(messages_bow,train_label)     # funtction for Logistic Regression 
Logistic_pred =LogisticRegressionClassifier.predict(bow_test)

print "Evaluation Metrics for NaiveBayes Classifier:"             # evaluation metrics for the classifiers 
print classification_report(test_label, Naive_predictions)
print "Evaluation Metrics for Logistic Regression  Classifier:"
print classification_report(test_label,Logistic_pred) 
