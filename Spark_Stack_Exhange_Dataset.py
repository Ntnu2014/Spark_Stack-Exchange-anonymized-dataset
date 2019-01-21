
# coding: utf-8

# # Spark with Stack Exchange anonymized data

# In[ ]:


# Spark is used for data manipulation, analysis, and machine learning on this data set.


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144


# In[ ]:


## Accessing the data


# In[ ]:


#There are three sub-folders, `allUsers`, `allPosts`, and `allVotes` 
#Gzipped XML 


# In[ ]:


get_ipython().system('aws s3 cp s3://mydata-course/spark-stats-data/stack_exchange_schema.txt .')


# In[ ]:


#You can either get the data by running the appropriate S3 commands in the terminal, 
#or by running this block for the smaller stats data set:


# In[4]:


get_ipython().system('mkdir -p spark-stats-data')
get_ipython().system("aws s3 sync --exclude '*' --include 'all*' s3://mydata-course/spark-stats-data/ ./spark-stats-data")
get_ipython().system("aws s3 sync --exclude '*' --include 'posts*zip' s3://mydata-course/spark-stats-data/ ./spark-stats-data")


# In[5]:


get_ipython().system('mkdir -p spark-stack-data')
get_ipython().system("aws s3 sync --exclude '*' --include 'all*' s3://mydata-course/spark-stack-data/ ./spark-stack-data")


# ## Data input and parsing_Bad XML
# 

# In[6]:


#Returning the total number XML rows that started with <row
from pyspark import SparkContext
sc = SparkContext("local[*]", "temp")
import os, time
def localpath(path):
    return 'file://' + os.path.join(os.path.abspath(os.path.curdir), path)

def isEntry(line):
    return "<row " in line

import re
def isValid(line):
    patt_row = re.compile('<row.*/>')
    return bool(re.search(patt_row, line))


# In[7]:


data_entries = sc.textFile(localpath('spark-stats-data/allPosts/'))         .filter(lambda x: isEntry(x))

valid_entries = sc.textFile(localpath('spark-stats-data/allPosts/'))         .filter(lambda x: isValid(x))


# ## Upvote percentage
# 

# In[10]:


#Calculate the average percentage of upvotes (upvotes / (upvotes + downvotes)) 
#for the smallest 50 keys.

from lxml import etree
valid_entries = sc.textFile(localpath('spark-stats-data/allVotes/'))         .filter(lambda x: isValid(x))
hm = valid_entries.take(4)


# In[11]:


class Post:
    def __init__(self, postId, favCount):
        self.postId = postId
        self.favCount = favCount

    @classmethod
    def parse(cls, line):
        postId = etree.fromstring(line).attrib['Id']
        if re.search('FavoriteCount', line):
            favCount = int(etree.fromstring(line).attrib['FavoriteCount'])
        else:
            favCount = 0
        return cls(postId, favCount)

postData = sc.textFile(localpath("spark-stats-data/allPosts"))     .filter(lambda x: isValid(x))     .map(Post.parse)

class Vote:
    def __init__(self, postId, voteType):
        self.postId = postId
        self.voteType = voteType

    @classmethod
    def parse(cls, line):
        postId = etree.fromstring(line).attrib['PostId']
        voteType = etree.fromstring(line).attrib['VoteTypeId']
        return cls(postId, voteType)

voteData = sc.textFile(localpath("spark-stats-data/allVotes"))     .filter(lambda x: isValid(x))     .map(Vote.parse)

def tupref(upvote):
    if upvote[1] == '2':
        return (upvote[0],(1,1))
    elif upvote[1] == '3':
        return (upvote[0],(0,1))
    else:
        return (upvote[0],(0,0))
def addtup(tup1,tup2):
    return (tup1[0]+tup2[0], tup1[1]+tup2[1])

a = postData.map(lambda p: (p.postId, p.favCount))     .join(voteData.map(lambda v: (v.postId, v.voteType)))     .map(lambda x: (x[1][0], x[1][1]))     .map(tupref)     .reduceByKey(addtup)     .map(lambda x: (x[0], x[1][0]/x[1][1]))     .sortByKey()     .take(50)


# In[12]:


a[:5]


# ## Answer percentage
# 

# In[14]:


#Investigate the correlation between a user's reputation and the kind of posts they make.
class PostQA:
    def __init__(self, UserId, PostTypeId):
        self.UserId = UserId
        self.PostTypeId = PostTypeId

    @classmethod
    def parse(cls, line):
        PostTypeId = etree.fromstring(line).attrib['PostTypeId']
        if re.search('OwnerUserId', line):
            UserId = etree.fromstring(line).attrib['OwnerUserId']
        else:
            UserId = '-9999'

        return cls(UserId, PostTypeId)

postQAData = sc.textFile(localpath("spark-stats-data/allPosts"))     .filter(lambda x: isValid(x))     .map(PostQA.parse)


class User:
    def __init__(self, UserId, Reputation):
        self.UserId = UserId
        self.Reputation = Reputation

    @classmethod
    def parse(cls, line):
        UserId = etree.fromstring(line).attrib['Id']
        Reputation = etree.fromstring(line).attrib['Reputation']
        return cls(UserId, Reputation)

userData = sc.textFile(localpath("spark-stats-data/allUsers"))     .filter(lambda x: isValid(x))     .map(User.parse)

def tupref(answer):
    if answer[1][1] == '2':
        return ((-int(answer[1][0]),answer[0]),(1,1))
    elif answer[1][1] == '1':
        return ((-int(answer[1][0]),answer[0]),(0,1))
    else:
        return ((-int(answer[1][0]),answer[0]),(0,0))
    
def addtup(tup1,tup2):
    return (tup1[0]+tup2[0], tup1[1]+tup2[1])

b = userData.map(lambda u: (u.UserId, u.Reputation))     .join(postQAData.map(lambda p: (p.UserId, p.PostTypeId)))     .map(tupref)     .reduceByKey(addtup)     .sortByKey()     .map(lambda x: (int(x[0][1]), x[1][0]/x[1][1]))     .take(99)


# In[15]:


s = 0
for i in b:
    s += i[1]
b.append((-1, s/len(b)))


# In[16]:


sc.textFile(localpath('spark-stats-data/allPosts/'))         .filter(lambda x: isValid(x))         .take(2)


# In[17]:


sc.textFile(localpath('spark-stats-data/allUsers/'))         .filter(lambda x: isValid(x))         .take(2)


# ## Post counts
# 

# In[19]:


#returning the top 100 post counts among all users (of all types of posts) 
#and the average reputation for every user who has that count.

class PostCount:
    def __init__(self, UserId, Id):
        self.UserId = UserId
        self.Id = Id

    @classmethod
    def parse(cls, line):
        Id = etree.fromstring(line).attrib['Id']
        if re.search('OwnerUserId', line):
            UserId = etree.fromstring(line).attrib['OwnerUserId']
        else:
            UserId = '-9999'
        return cls(UserId, Id)

postCountData = sc.textFile(localpath("spark-stats-data/allPosts"))     .filter(lambda x: isValid(x))     .map(PostCount.parse)

class User:
    def __init__(self, UserId, Reputation):
        self.UserId = UserId
        self.Reputation = Reputation

    @classmethod
    def parse(cls, line):
        UserId = etree.fromstring(line).attrib['Id']
        Reputation = etree.fromstring(line).attrib['Reputation']
        return cls(UserId, Reputation)

userData = sc.textFile(localpath("spark-stats-data/allUsers"))     .filter(lambda x: isValid(x))     .map(User.parse)

c = postCountData.map(lambda p: (p.UserId, p.Id))     .join(userData.map(lambda u: (u.UserId, u.Reputation)))     .map(lambda x: ((x[0], x[1][1]),1))     .reduceByKey(lambda x, y : x + y)     .map(lambda x: (x[1], int(x[0][1])))     .aggregateByKey((0,0),lambda a,b: (a[0] + b,  a[1] + 1),
                           lambda a,b: (a[0] + b[0], a[1] + b[1])) \
    .mapValues(lambda v: v[0]/v[1]) \
    .sortByKey(ascending= False) \
    .take(100)


# ## Quick answers
# 

# In[22]:


# Returning a list, whose i'th element correspond to i'th hour

class QuestionCount:
    def __init__(self, aId, qId, PostTypeId, qCreationDate):
        self.qId = qId
        self.aId = aId
        self.PostTypeId = PostTypeId
        self.qCreationDate = qCreationDate

    @classmethod
    def parse(cls, line):
        qId = etree.fromstring(line).attrib['Id']
        PostTypeId = etree.fromstring(line).attrib['PostTypeId']
        qCreationDate = etree.fromstring(line).attrib['CreationDate']
        if re.search('AcceptedAnswerId', line):
            aId = etree.fromstring(line).attrib['AcceptedAnswerId']
        else:
            aId = '-999'
        return cls(aId, qId, PostTypeId, qCreationDate)
    
QuestionData = sc.textFile(localpath("spark-stats-data/allPosts"))     .filter(lambda x: isValid(x))     .map(QuestionCount.parse)

class AnsweredCount:
    def __init__(self, aId, aType, aCreationDate):
        self.aId = aId
        self.aType = aType
        self.aCreationDate = aCreationDate

    @classmethod
    def parse(cls, line):
        aId = etree.fromstring(line).attrib['Id']
        aType = etree.fromstring(line).attrib['PostTypeId']
        aCreationDate = etree.fromstring(line).attrib['CreationDate']      
        return cls(aId, aType, aCreationDate)
    
AnweredData = sc.textFile(localpath("spark-stats-data/allPosts"))     .filter(lambda x: isValid(x))     .map(AnsweredCount.parse)


# In[23]:


from datetime import datetime
def onlyQA(x):
    if x[1][1]=='1' and x[1][3]=='2':
        return True
    else:
        return False
    
def quickie(x):
    qT = datetime.strptime(x[1][0], '%Y-%m-%dT%H:%M:%S.%f')
    aT = datetime.strptime(x[1][1], '%Y-%m-%dT%H:%M:%S.%f')
    h_gap = (aT-qT).total_seconds()/(60*60)
    return h_gap < 3


quick_by_hour = QuestionData.map(lambda q: (q.aId, (q.qId, q.PostTypeId, q.qCreationDate)))     .join(AnweredData.map(lambda a: (a.aId, (a.aType, a.aCreationDate))))     .map(lambda x: (x[0], x[1][0] + x[1][1]))     .filter(lambda x: onlyQA(x))     .map(lambda x: (x[0],(x[1][2],x[1][4])))     .filter(quickie)     .map(lambda x: (datetime.strptime(x[1][0], '%Y-%m-%dT%H:%M:%S.%f').hour, 1))     .reduceByKey(lambda x, y: x + y)     .collect()

totAnswered_by_hour = QuestionData.map(lambda q: (q.aId, (q.qId, q.PostTypeId, q.qCreationDate)))     .join(AnweredData.map(lambda a: (a.aId, (a.aType, a.aCreationDate))))     .map(lambda x: (x[0], x[1][0] + x[1][1]))     .filter(lambda x: onlyQA(x))     .map(lambda x: (x[0],(x[1][2],x[1][4])))     .map(lambda x: (datetime.strptime(x[1][0], '%Y-%m-%dT%H:%M:%S.%f').hour, 1))     .reduceByKey(lambda x, y: x + y)     .collect()


# In[24]:


sorted_short = [i[1] for i in sorted(quick_by_hour, key = lambda x: x[0])]
sorted_tot = [i[1] for i in sorted(totAnswered_by_hour, key = lambda x: x[0])]


# In[25]:


from operator import truediv
d = [truediv(*x) for x in zip(sorted_short, sorted_tot)]


# ## Quick answers&mdash;full
# 

# In[ ]:


#Returning a list, whose i'th element correspond to i'th hour 
#on the full Stack Exchange data set.


# In[27]:



line = '  <row Body="See `continuous-data`" CommentCount="0" CreationDate="2013-10-28T10:42:29.940" Id="73934" LastActivityDate="2013-10-28T10:42:29.940" LastEditDate="2013-10-28T10:42:29.940" LastEditorUserId="686" OwnerUserId="686" PostTypeId="4" Score="0" />'
parsedline = etree.fromstring(line)
'PostTypeId' in parsedline.attrib


# In[28]:


class QuestionCount:
    def __init__(self, aId, qId, PostTypeId, qCreationDate):
        self.qId = qId
        self.aId = aId
        self.PostTypeId = PostTypeId
        self.qCreationDate = qCreationDate

    @classmethod
    def parse(cls, line):
        parsedline = etree.fromstring(line)
        qId = parsedline.attrib['Id']
        PostTypeId = parsedline.attrib['PostTypeId']
        qCreationDate = parsedline.attrib['CreationDate']
        if 'AcceptedAnswerId' in parsedline.attrib:
            aId = parsedline.attrib['AcceptedAnswerId']
        else:
            aId = '-999'
        return cls(aId, qId, PostTypeId, qCreationDate)
    
QuestionData = sc.textFile(localpath("spark-stack-data/allPosts"))     .filter(lambda x: isValid(x))     .map(QuestionCount.parse)

class AnsweredCount:
    def __init__(self, aId, aType, aCreationDate):
        self.aId = aId
        self.aType = aType
        self.aCreationDate = aCreationDate

    @classmethod
    def parse(cls, line):
        parsedline = etree.fromstring(line)
        aId = parsedline.attrib['Id']
        aType = parsedline.attrib['PostTypeId']
        aCreationDate = parsedline.attrib['CreationDate']      
        return cls(aId, aType, aCreationDate)
    
AnweredData = sc.textFile(localpath("spark-stack-data/allPosts"))     .filter(lambda x: isValid(x))     .map(AnsweredCount.parse)


# In[29]:


def onlyQA(x):
    if x[1][1]=='1' and x[1][3]=='2':
        return True
    else:
        return False
    
def quickie(x):
    qT = datetime.strptime(x[1][0], '%Y-%m-%dT%H:%M:%S.%f')
    aT = datetime.strptime(x[1][1], '%Y-%m-%dT%H:%M:%S.%f')
    h_gap = (aT-qT).total_seconds()/(60*60)
    return h_gap < 3


# In[30]:


quick_by_hour = QuestionData.map(lambda q: (q.aId, (q.qId, q.PostTypeId, q.qCreationDate)))     .join(AnweredData.map(lambda a: (a.aId, (a.aType, a.aCreationDate))))     .map(lambda x: (x[0], x[1][0] + x[1][1]))     .filter(lambda x: onlyQA(x))     .map(lambda x: (x[0],(x[1][2],x[1][4])))     .filter(quickie)     .map(lambda x: (datetime.strptime(x[1][0], '%Y-%m-%dT%H:%M:%S.%f').hour, 1))     .reduceByKey(lambda x, y: x + y)     .collect()

totAnswered_by_hour = QuestionData.map(lambda q: (q.aId, (q.qId, q.PostTypeId, q.qCreationDate)))     .join(AnweredData.map(lambda a: (a.aId, (a.aType, a.aCreationDate))))     .map(lambda x: (x[0], x[1][0] + x[1][1]))     .filter(lambda x: onlyQA(x))     .map(lambda x: (x[0],(x[1][2],x[1][4])))     .map(lambda x: (datetime.strptime(x[1][0], '%Y-%m-%dT%H:%M:%S.%f').hour, 1))     .reduceByKey(lambda x, y: x + y)     .collect()


# In[31]:


sorted_short = [i[1] for i in sorted(quick_by_hour, key = lambda x: x[0])]
sorted_tot = [i[1] for i in sorted(totAnswered_by_hour, key = lambda x: x[0])]
e = [truediv(*x) for x in zip(sorted_short, sorted_tot)]


# ## Identify veterans
# 

# In[33]:


# Identifying "veteran_score","veteran_views","vet_favorites",
# "vet_answers","brief_score","brief_views","brief_answers","brief_favorites"


class postCount:
    def __init__(self, UserId, pCreationDate, PostTypeId, Score, ViewCount, AnswerCount, FavoriteCount):
        self.UserId = UserId
        self.pCreationDate = pCreationDate
        self.PostTypeId = PostTypeId
        self.Score = Score
        self.ViewCount = ViewCount
        self.AnswerCount = AnswerCount
        self.FavoriteCount = FavoriteCount

    @classmethod
    def parse(cls, line):
        parsedline = etree.fromstring(line)
        pCreationDate = parsedline.attrib['CreationDate']
        PostTypeId = parsedline.attrib['PostTypeId']
        
        UserId = parsedline.attrib.get('OwnerUserId','-999')
        Score = float(parsedline.attrib.get('Score',0))
        ViewCount = float(parsedline.attrib.get('ViewCount',0))
        AnswerCount = float(parsedline.attrib.get('AnswerCount',0))
        FavoriteCount = float(parsedline.attrib.get('FavoriteCount',0))
            
        return cls(UserId, pCreationDate, PostTypeId, Score, ViewCount, AnswerCount, FavoriteCount)

postCountData = sc.textFile(localpath("spark-stats-data/allPosts"))     .filter(lambda x: isValid(x))     .map(postCount.parse)

class User:
    def __init__(self, UserId, uCreationDate):
        self.UserId = UserId
        self.uCreationDate = uCreationDate

    @classmethod
    def parse(cls, line):
        parsedline = etree.fromstring(line)
        UserId = parsedline.attrib['Id']
        uCreationDate = parsedline.attrib['CreationDate']
        return cls(UserId, uCreationDate)

userData = sc.textFile(localpath("spark-stats-data/allUsers"))     .filter(lambda x: isValid(x))     .map(User.parse)


def vetPotential(x):
    uT = datetime.strptime(x[1][0], '%Y-%m-%dT%H:%M:%S.%f')
    pT = datetime.strptime(x[1][1], '%Y-%m-%dT%H:%M:%S.%f')
    d_gap = (pT-uT).days
    return int(d_gap < 150 and d_gap >= 100)


# In[34]:


def isQuest(x):
    return x[1][0]=='1'

vedIds = userData.map(lambda u: (u.UserId, u.uCreationDate))     .join(postCountData.map(lambda p: (p.UserId, p.pCreationDate)))     .map(lambda x: (x[0], vetPotential(x)))     .reduceByKey(lambda x, y: x + y)


# In[35]:


vedIds.filter(lambda x: x[1] == 0).count()


# In[36]:


def Quest_1(x, y):
    if datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S.%f') < datetime.strptime(y[1], '%Y-%m-%dT%H:%M:%S.%f'):
        return x
    else:
        return y


# In[37]:


vetTuple = vedIds.filter(lambda x: x[1] > 0)     .join(postCountData.map(lambda p: (p.UserId, (p.PostTypeId, p.pCreationDate, p.Score, p.ViewCount, p.AnswerCount, p.FavoriteCount))))     .map(lambda x: (x[0], x[1][1]))     .filter(lambda x: isQuest(x))     .reduceByKey(Quest_1)     .map(lambda x : (float(x[1][2]), float(x[1][3]), float(x[1][4]), float(x[1][5]), 1))     .reduce(lambda x ,y : (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4]))


# In[38]:


vetlist = list(vetTuple)
[i/vetlist[-1] for i in vetlist[:-1]]


# In[39]:


briefTuple = vedIds.filter(lambda x: x[1] == 0)     .join(postCountData.map(lambda p: (p.UserId, (p.PostTypeId, p.pCreationDate, p.Score, p.ViewCount, p.AnswerCount, p.FavoriteCount))))     .map(lambda x: (x[0], x[1][1]))     .filter(lambda x: isQuest(x))     .reduceByKey(Quest_1)     .map(lambda x : (float(x[1][2]), float(x[1][3]), float(x[1][4]), float(x[1][5]), 1))     .reduce(lambda x ,y : (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4]))
brieflist = list(briefTuple)
[i/brieflist[-1] for i in brieflist[:-1]]


# ## Identify veterans&mdash;full
# 

# In[41]:


# Identifying veterans on full Stack Exchange data set

class postCount:
    def __init__(self, UserId, pCreationDate, PostTypeId, Score, ViewCount, AnswerCount, FavoriteCount):
        self.UserId = UserId
        self.pCreationDate = pCreationDate
        self.PostTypeId = PostTypeId
        self.Score = Score
        self.ViewCount = ViewCount
        self.AnswerCount = AnswerCount
        self.FavoriteCount = FavoriteCount

    @classmethod
    def parse(cls, line):
        parsedline = etree.fromstring(line)
        pCreationDate = parsedline.attrib['CreationDate']
        PostTypeId = parsedline.attrib['PostTypeId']
        
        UserId = parsedline.attrib.get('OwnerUserId','-999')
        Score = float(parsedline.attrib.get('Score',0))
        ViewCount = float(parsedline.attrib.get('ViewCount',0))
        AnswerCount = float(parsedline.attrib.get('AnswerCount',0))
        FavoriteCount = float(parsedline.attrib.get('FavoriteCount',0))
            
        return cls(UserId, pCreationDate, PostTypeId, Score, ViewCount, AnswerCount, FavoriteCount)

postCountData = sc.textFile(localpath("spark-stack-data/allPosts"))     .filter(lambda x: isValid(x))     .map(postCount.parse)

class User:
    def __init__(self, UserId, uCreationDate):
        self.UserId = UserId
        self.uCreationDate = uCreationDate

    @classmethod
    def parse(cls, line):
        parsedline = etree.fromstring(line)
        UserId = parsedline.attrib['Id']
        uCreationDate = parsedline.attrib['CreationDate']
        return cls(UserId, uCreationDate)

userData = sc.textFile(localpath("spark-stack-data/allUsers"))     .filter(lambda x: isValid(x))     .map(User.parse)


# In[42]:


def vetPotential(x):
    uT = datetime.strptime(x[1][0], '%Y-%m-%dT%H:%M:%S.%f')
    pT = datetime.strptime(x[1][1], '%Y-%m-%dT%H:%M:%S.%f')
    d_gap = (pT-uT).days
    return int(d_gap < 150 and d_gap >= 100)

def isQuest(x):
    return x[1][0]=='1'

def Quest_1(x, y):
    if datetime.strptime(x[1], '%Y-%m-%dT%H:%M:%S.%f') < datetime.strptime(y[1], '%Y-%m-%dT%H:%M:%S.%f'):
        return x
    else:
        return y
    
vedIds = userData.map(lambda u: (u.UserId, u.uCreationDate))     .join(postCountData.map(lambda p: (p.UserId, p.pCreationDate)))     .map(lambda x: (x[0], vetPotential(x)))     .reduceByKey(lambda x, y: x + y)


# In[43]:


vetTuple = vedIds.filter(lambda x: x[1] > 0)     .join(postCountData.map(lambda p: (p.UserId, (p.PostTypeId, p.pCreationDate, p.Score, p.ViewCount, p.AnswerCount, p.FavoriteCount))))     .map(lambda x: (x[0], x[1][1]))     .filter(lambda x: isQuest(x))     .reduceByKey(Quest_1)     .map(lambda x : (float(x[1][2]), float(x[1][3]), float(x[1][4]), float(x[1][5]), 1))     .reduce(lambda x ,y : (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4]))

vetlist = list(vetTuple)
[i/vetlist[-1] for i in vetlist[:-1]]


# In[44]:


briefTuple = vedIds.filter(lambda x: x[1] == 0)     .join(postCountData.map(lambda p: (p.UserId, (p.PostTypeId, p.pCreationDate, p.Score, p.ViewCount, p.AnswerCount, p.FavoriteCount))))     .map(lambda x: (x[0], x[1][1]))     .filter(lambda x: isQuest(x))     .reduceByKey(Quest_1)     .map(lambda x : (float(x[1][2]), float(x[1][3]), float(x[1][4]), float(x[1][5]), 1))     .reduce(lambda x ,y : (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4]))
brieflist = list(briefTuple)
[i/brieflist[-1] for i in brieflist[:-1]]


# ## Word2vec
# 

# In[46]:


# Using an alternative approach for vectorizing text data 
# for predicting other words in the document

def isValid(line):
    patt_row = re.compile('<row.*/>')
    return bool(re.search(patt_row, line))

def hasTag(x):
    return x[1] != '-999'

class postCount:
    def __init__(self, Id, Tags):
        self.Id = Id
        self.Tags = Tags

    @classmethod
    def parse(cls, line):
        parsedline = etree.fromstring(line)
        Id = parsedline.attrib['Id']
        
        if 'Tags' in parsedline.attrib:
            Tags = parsedline.attrib['Tags']
        else:
            Tags = '-999'            
        return cls(Id, Tags)


# In[47]:


from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

from pyspark.ml.feature import Word2Vec

posts = sc.textFile(localpath("spark-stack-data/allPosts"))     .filter(lambda x: isValid(x))     .map(postCount.parse)     .map(lambda p: (p.Id, p.Tags))     .filter(lambda x: hasTag(x))     .map(lambda x: x[1])     .map(lambda line: (re.findall(r'<(.*?)>',line),1))     .toDF(['Tags', 'score'])


w2v = Word2Vec(inputCol="Tags", outputCol="vectors", vectorSize=100, seed=42)
model = w2v.fit(posts)
result = model.transform(posts)

vectors = model.getVectors().rdd.map(lambda x: (x.word, x.vector))
list_rows = model.findSynonyms('ggplot2', 25).rdd.map(lambda entry: (entry['word'], float(entry['similarity']))).collect()


# In[48]:


posts.head()


# ## Classification
# 

# In[66]:


# predicting the tags of a question from its body text using logistic regression

class tagStuff:
    def __init__(self, Tags, Body):
        self.Tags = Tags
        self.Body = Body

    @classmethod
    def parse(cls, line):
        parsedline = etree.fromstring(line)
        Tags = parsedline.attrib.get('Tags','-999')
        Body = parsedline.attrib.get('Body','-999')            
        return cls(Tags, Body)

def hasTag(x):
    return x[0] != '-999'


# In[67]:


tag_count = sc.textFile(localpath("spark-stats-data/training"))     .filter(lambda x: isValid(x))     .map(tagStuff.parse)     .map(lambda p: (p.Tags, p.Body))     .filter(lambda x: hasTag(x))     .map(lambda line: (re.findall(r'<(.*?)>',line[0]),line[1]))     .flatMap(lambda x: [(i, x[1]) for i in x[0]])


# In[68]:


tag_count_dict = tag_count.countByKey()
tag_100 = [tag[0] for tag in sorted(tag_count_dict.items(), key = lambda x: x[1], reverse = True)[:100]]


# In[70]:


from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
test_tag_count = sc.textFile(localpath("spark-stats-data/test"))     .filter(lambda x: isValid(x))     .map(tagStuff.parse)     .map(lambda p: (p.Tags, p.Body))     .filter(lambda x: hasTag(x))

test_data = sqlContext.createDataFrame(test_tag_count.map(lambda x: (x[1], x[0])), ["Body", "label"])


# In[71]:


sc.textFile(localpath("spark-stats-data/test"))     .filter(lambda x: isValid(x))     .map(tagStuff.parse)     .map(lambda p: (p.Tags, p.Body))     .filter(lambda x: hasTag(x))     .count()


# In[72]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

tokenizer = Tokenizer(inputCol="Body", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
logreg = LogisticRegression(maxIter=10, regParam=0.01)

pipeline = Pipeline(stages=[tokenizer, hashingTF, logreg])


# In[73]:


r_training_data = sqlContext.createDataFrame(tag_count.map(lambda x: (x[1], int(x[0] == 'r'))), ["Body", "label"])
model = pipeline.fit(r_training_data)


# In[74]:


tag_100[:5]


# In[75]:


tag_probs = []

for tags in tag_100:
    training = sqlContext.createDataFrame(tag_count.map(lambda x: (x[1], int(x[0] == tags))), ["Body", "label"]).cache()
    model = pipeline.fit(training)
    predictions = model.transform(test_data).select("probability").collect()
    pred_list = [i[0][0] for i in predictions]
    tag_probs.append((tags,pred_list))
    training.unpersist()

