import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk,re,pprint
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.datasets import load_files
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
stemmer = SnowballStemmer("english")
data =load_files('C:/Users/Abayomi/Desktop/cluster/', description=None, categories=('AlexanderSmith','BenjaminKangLim','BernardHickey','AaronPressman','AlanCrosby'), load_content=True, shuffle=True, encoding='UTF-8', decode_error='strict', random_state=0)
def tokenize_and_stem(data):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(data) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
def tokenize_only(data):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(data) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:                # filter removing regular expression 
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in data['data']:
    allwords_stemmed = tokenize_and_stem(i) 
    totalvocab_stemmed.extend(allwords_stemmed) 
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

bow_transformer = CountVectorizer(stop_words='english',tokenizer=tokenize_and_stem,analyzer=u'word' ).fit(data['data'])

#data_bow = bow_transformer.transform(data['data'])
#tfidf_transformer = TfidfTransformer().fit(data['data'])
#messages_tfidf = tfidf_transformer.transform(data['data'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(data['data']) #fit the vectorizer 

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()  # TF-IDF feature extraction method 
#terms=bow_transformer.get_feature_names()     # Bag of words feature extraction method
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=20000000, n_init=1)
model.fit(tfidf_matrix)
clusters = model.labels_.tolist()

true_label = data['target']
doc_file = data['filenames']

#doc_label = training_data['target_names']

doc_label = []
t=0
u=0
for i in range(len(doc_file)):
#if t<=50:
    if true_label[i] <= 0:
        t=t+1
        q='AP'+str(t)
        doc_label.append(q)
    else:
        u=u+1
        q='AC'+str(u)
        doc_label.append(q)
dist = 1 - cosine_similarity(tfidf_matrix)

print
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
#terms = vectorizer.get_feature_names()
#terms = tfidf_vectorizer.get_feature_names()

for i in range(num_clusters):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print

docs = { 'label': doc_label, 'documents': doc_file, 'cluster': clusters }
frame = pd.DataFrame(docs, index = [clusters] , columns = ['label', 'cluster'])
print 
print frame['cluster'].value_counts()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
cluster_colors = {0: '#1b9e77', 1: '#d95f02',2:'#7570b3', 3: '#e7298a', 4: '#66a61e'}
cluster_names = {0: 'cluster 0', 1: 'cluster 1',2:'cluster 2',3:'cluster 3',4:'cluster 4'}
#create data frame that has the result of the MDS plus the cluster numbers and labels
df = pd.DataFrame(dict(x=xs, y=ys, doc_cluster=clusters, label=doc_label)) 

#group by cluster
groups = df.groupby('doc_cluster')
# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) 
ax.margins(0.05) 
#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['label'], size=8)  

plt.show()
pred_label = []
pred = []
for i in frame['cluster']:
    pred.append(i)
for i in pred:
    if pred[0] == 0:
        pred_label.append(i)
    else: pred_label.append(1-i)
print
print 'Clustering Accuracy = '+ str(accuracy_score(true_label, pred_label))