
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] =15,9
import findspark

import glob,csv
import pandas as pd
df = pd.concat([pd.read_csv(f, encoding='latin1', quoting=csv.QUOTE_NONE,error_bad_lines=False)                 for f in glob.glob(r'C:\Users\Lalith Chandra A\BDP_Project\temp_csv\*.csv')])


from pyspark.sql import SparkSession, column
spark = SparkSession.builder.appName("Hive TST").enableHiveSupport().getOrCreate()

spark

spark_df=spark.createDataFrame(df)

spark_df.show()

spark_df.createOrReplaceTempView("streamed_tweets")


import findspark

findspark.init('C:\spark-2.4.5-bin-hadoop2.7')

# May cause deprecation warnings, safe to ignore, they aren't errors
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import desc
from pyspark.sql import Row
import re

# Can only run this once. restart your kernel for any errors.
sc = SparkContext.getOrCreate()
# sc1 = SparkSession.builder.appName("Hive TST").enableHiveSupport().getOrCreate()



ssc = StreamingContext(sc, 10 )
sqlContext = SQLContext(sc)
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.hive.HiveContext
val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
df.head(20)
((df[~(df['_1'].str.contains('[A-Za-z]'))].count()[0])/df.count()[0])*100
df.rename(columns={'_1':'tweet_txt'},inplace=True)
df.head()
df.dtypes
import re
def data_cleansing(corpus):
    letters_only = re.sub("[^a-zA-Z]", " ", corpus) 
    words = letters_only.lower().split()                            
    return( " ".join( words ))
df['tweet_txt'] = df['tweet_txt'].apply(lambda x:data_cleansing(x))
df.head()

from textblob import TextBlob
df['sentiment_value']=df.tweet_txt.apply(lambda x:TextBlob(str((x).encode('ascii', 'ignore'))).sentiment.polarity)
df['sentiment_score']=np.where(df.sentiment_value<=0.0,1,0)
df['sentiment_description']=np.where(df.sentiment_value<=0.0,'negative','positive')
df.head(30)
print(df.sentiment_score.value_counts(),'\n\n',df.sentiment_description.value_counts())
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def wordcloud(source,stop):
    tmp = df[df['sentiment_description']==source]
    clean_text=[]
    for each in tmp['tweet_txt']:
        clean_text.append(each)
    clean_text = ' '.join(clean_text)
    if source == 'positive' :
        color='white'
    else:
        color='black'
    if (stop=="yes"):    
        wordcloud = WordCloud(background_color=color,
                          width=3500,
                          height=3000,stopwords = stopwords
                         ).generate(clean_text)
    else:
        wordcloud = WordCloud(background_color=color,
                          width=3500,
                          height=3000
                         ).generate(clean_text)
    print('==='*30)
    print('word cloud of '+source+' is plotted below')
    plt.figure(1,figsize=(8,8))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()

stopwords.add('co')
stopwords.add('https')
stopwords.add('hey')
stopwords.add('hello')
stopwords.add('school')
wordcloud('positive',"yes")
wordcloud('negative',"yes")

from sklearn.model_selection import train_test_split
train, test = train_test_split(df,test_size=0.3)
df.head()
train_corpus = []
test_corpus = []
for each in train['tweet_txt']:
    train_corpus.append(each)
for each in test['tweet_txt']:
    test_corpus.append(each)
## Start creating them
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(stop_words='english',strip_accents='unicode',
                                    token_pattern=r'\w{2,}')
train_features = v.fit_transform(train_corpus)
test_features=v.transform(test_corpus)
print(train_features.shape)
print(test_features.shape)
v.get_feature_names()
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
Classifiers = {'lg':LogisticRegression(random_state=42,C=5,max_iter=200),               'dt':DecisionTreeClassifier(random_state=42,min_samples_leaf=1),               'rf':RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=-1),               'gb':GradientBoostingClassifier(random_state=42,n_estimators=100,learning_rate=0.3)}

def ML_Pipeline(clf_name):
    clf = Classifiers[clf_name]
    fit = clf.fit(train_features,train['sentiment_description'])
    pred = clf.predict(test_features)
    Accuracy = accuracy_score(test['sentiment_description'],pred)
    Confusion_matrix = confusion_matrix(test['sentiment_description'],pred)
    print('==='*35)
    print('Accuracy of '+ clf_name +' is '+str(Accuracy))
    print('==='*35)
    print(Confusion_matrix)
ML_Pipeline('lg')
ML_Pipeline('dt')
test_corpus
train['sentiment_description']

clf = RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=-1)
fit = clf.fit(train_features,train['sentiment_description'])

words = v.get_feature_names()
importance = clf.feature_importances_
impordf = pd.DataFrame({'Word' : words,'Importance' : importance})
impordf = impordf.sort_values(['Importance', 'Word'], ascending=[0, 1])
impordf.head(20)
clf = LogisticRegression(random_state=42,C=5,max_iter=200)
fit = clf.fit(train_features,train['sentiment_description'])
pred = clf.predict(test_features)
Accuracy = accuracy_score(test['sentiment_description'],pred)
Confusion_matrix = confusion_matrix(test['sentiment_description'],pred)
print('==='*35)
print('Accuracy of '+ 'lr' +' is '+str(Accuracy))
print('==='*35)
print(Confusion_matrix)
get_ipython().system('pip install joblib')
from joblib import dump,load
dump(fit,'lr.joblib')
dump(v,'tfid.joblib')

from joblib import dump,load
model=load('lr.joblib')
tfidf_temp=load('tfid.joblib')

a=["internet is very slow",
   "issue with billing",
   "nice service provided",
   "thanks problem resolved"
   ,"HeyFriends I found great iPhone 6S giveaway     you can get it here---> #iphone6Sgiveawy2k16     Check it out looks like great freebie.     Don't Drop This"
  ,"Unfortunately I will have to transfer my lines to a different provider.  I would like to ensure there won't be any issues when my new provider attempts to port my numbers to their service?"]
tdf=pd.Series(a).astype(str).apply(lambda x:data_cleansing(x))
t_a=tfidf_temp.transform(tdf)
pred1 = model.predict(t_a)
pred1




