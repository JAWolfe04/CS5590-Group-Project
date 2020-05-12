#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

socket_stream = ssc.socketTextStream("192.168.1.239", 5551)

lines = socket_stream.window( 20,20 )

from collections import namedtuple
fields = ("text", "location","source" )
Tweet = namedtuple( 'Tweet', fields )

fields = ("tag", "count" )
Tweet = namedtuple( 'Tweet', fields )

# lines.map(lambda x:x.toString).foreachRDD(lambda rddRaw : rddRaw.toDF()).limit(10).registerTempTable("tweets1")


( lines.map( lambda text: text.split( "~@" ) ) #Splits to a list
  .foreachRDD( 
      lambda rdd: rdd.toDF().registerTempTable("tweets") 
  )) # Registers to a table.

( lines.flatMap( lambda text: text.split( " " ) ) 
  .filter( lambda word: word.lower().startswith("#") ) 
  .map( lambda word: ( word.lower(), 1 ) ) 
  .reduceByKey( lambda a, b: a + b ) 
  .map( lambda rec: Tweet( rec[0], rec[1] ) ) 
  .foreachRDD( lambda rdd: rdd.toDF().sort( desc("count") ) 
  .limit(10).registerTempTable("tweets1") ) )

lines.map(lambda x: (x.replace(',','').lower(), )).foreachRDD( lambda rdd: rdd.toDF().registerTempTable("tweets2") )

def Find(string): 
    # findall() has been used  
    # with valid conditions for urls in string 
    return (re.search("(?P<url>https?://[^\s]+)", string))

lines.flatMap(lambda text: text.split(" ")).filter( lambda word: word.startswith("http")).map( lambda word: ( word.lower(), 1 ) ).reduceByKey( lambda a, b: a + b ).foreachRDD( lambda rdd: rdd.toDF()  .limit(10).registerTempTable("tweets_url") )

lines.pprint()


# In[2]:



import time
from IPython import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
# Only works for Jupyter Notebooks!
get_ipython().run_line_magic('matplotlib', 'inline')

ssc.start()


# In[4]:


from pyspark.sql import SparkSession, column
spark = SparkSession.builder.appName("Hive TST").enableHiveSupport().getOrCreate()


# In[5]:


import re,pandas as pd
from joblib import dump,load
from textblob import TextBlob


model=load('lr.joblib')
tfidf_temp=load('tfid.joblib')


def data_cleansing(corpus):
    letters_only = re.sub("[^a-zA-Z]", " ", corpus) 
    words = letters_only.lower().split()                            
    return( " ".join( words ))

def model_predict(text):
    t_a=tfidf_temp.transform(pd.Series(text).astype(str))
    pred1 = model.predict(t_a)
    print(pred1)
    return (pred1[0])

def model_textblob(text):
    score=TextBlob(str((text).encode('ascii', 'ignore'))).sentiment.polarity
    if score <= 0.0 :
        return ('negative')
    else:
        return ('positive')

from pyspark.sql.types import StringType
spark.udf.register("custom_udf", data_cleansing, StringType())
spark.udf.register("custom_predict",model_predict,StringType())
spark.udf.register("texblob",model_textblob,StringType())


# In[6]:


spark.sql("SELECT _1,custom_predict(custom_udf(_1)),texblob(custom_udf(_1)) FROM tweets2").show()


# In[4]:


count = 0
while count < 10:
    time.sleep(5)
    sqlContext.sql( 'Select _1 url,_2 ct from tweets_url').show(truncate=False)
    display.clear_output(wait=True)
    count = count + 1


# In[ ]:


count = 0
while count < 10:
    time.sleep(5)
    top_10_tweets = sqlContext.sql( 'Select tag, count from tweets1' )
    top_10_df = top_10_tweets.toPandas()
    display.clear_output(wait=True)
    plt.figure( figsize = ( 10, 8 ) )
    sns.barplot( x="count", y="tag", data=top_10_df)
    plt.show()
    count = count + 1


# In[ ]:


count = 0
while count < 10:
    
    time.sleep( 5 )
    df=sqlContext.sql( "Select 'negative' tag,count(distinct _1) ct from tweets2 WHERE (_1 LIKE '%suffered%' or _1 LIKE '%killed%' or _1 LIKE '%deaths%' or _1 LIKE '%disappoi%' or _1 LIKE '%sad%' or _1 LIKE '%concern%' or _1 LIKE '%bad%' or _1 LIKE '%fail%')                UNION                Select 'positive' tag,count(distinct _1) ct from tweets2 WHERE (_1 LIKE '%survived%' or _1 LIKE '%happy%' OR _1 LIKE '%wonderful%' OR _1 LIKE '%bliss%' OR _1 LIKE '%hope%' OR _1 LIKE '%win%')").toPandas()
    display.clear_output(wait=True)
    plt.figure( figsize = ( 10, 8 ) )
    sns.set(style="whitegrid")
    sns.barplot( x="tag", y="ct", data=df)
    plt.show()
    count = count + 1


# In[4]:


#Displaying Sources Statistics
count = 0
while count < 1:
    
    time.sleep( 5 )
    
    top_10_source = sqlContext.sql('select source,count(*) as count from(select regexp_extract(tweets._3,\'Android|Twitter Web App|iPhone|Facebook|HubSpot|iPad|EcoInternet3|Echofon\',0) as source from tweets) where nullif(source,"") is not null group by source' )
    top_10_df = top_10_source.toPandas()
    display.clear_output(wait=True)
    plt.figure( figsize = ( 10, 8 ) )
    sns.barplot( x="count", y="source", data=top_10_df)
    plt.show()
    
    count = count + 1


# In[6]:


#Dispalying Location Statistics
count = 0
while count < 1:
    
    time.sleep( 5 )
    top_10_source = sqlContext.sql( 'select tweets._2 as location,count(*) as count from tweets where nullif(tweets._2,"") is not null and tweets._2 not like "%None%" group by tweets._2' )
    top_10_source.show()
    top_10_df = top_10_source.toPandas()
    display.clear_output(wait=True)
    plt.figure( figsize = ( 10, 8 ) )
    sns.barplot( x="count", y="location", data=top_10_df)
    plt.show()
    
    count = count + 1


# In[ ]:


ssc.stop()


# In[ ]:





# In[ ]:




