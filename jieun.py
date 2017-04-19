
# coding: utf-8

# In[32]:

import os # 파일을 열기위해
import urllib.request #Crawling
import bs4 #Crawling
import nltk #Pre-processing 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer #Pre-processing 
from string import punctuation #Pre-processing 
import numpy as np #classification
import math #classification
from matplotlib import pyplot as plt # visualization 
import sys


# In[33]:

training_files = ['category/entertainment_20.txt', 'category/golf_20.txt', 'category/politics_20.txt']
#test_files = ['document/entertainment.txt', 'document/golf.txt', 'document/politics.txt']


# In[34]:

def get_stemmedTokens_fromUrl(url):
    
    #Crawling(BeautifulSoup)
    htmlData = urllib.request.urlopen(url.strip())
    bs = bs4.BeautifulSoup(htmlData,'lxml')

    ##헤드라인 크롤링
    headlines=bs.findAll('h1','pg-headline')
    headline = ''
    for h in headlines:
        headline += h.getText()

    ##BODY 크롤링
    ##파싱한 데이터 중 <div class="zn-body__paragraph">를 find_all()로 전부 찾는다.
    bodies=bs.findAll('div','zn-body__paragraph')
    body = ''
    for b in bodies:
        body += b.getText()

    ##BODY 중에서 첫문단 크롤링 - 위의 BODY크롤링에서는 첫문단이 빠지고 크롤링된다.
    ##모든 기사의 첫 문단마다 반복되는 (CNN)이라는 글자는 크롤링 해오지 않도록 하였다.
    if bs.find('cite', class_="el-editorial-source"):
        bs.find('cite', class_="el-editorial-source").decompose() 
    para1=bs.findAll('div','el__leafmedia el__leafmedia--sourced-paragraph')
    first_para = ''
    for p in para1:
        first_para += p.getText()

    ##이미지캡션 크롤링
    captions = bs.findAll('div','media__caption el__gallery_image-title')
    caption = ''
    for c in captions:
        caption += c.getText()

    ##Pre-processing 
    #앞에서 크롤링 한 것을 모두 합쳐서 전처리할 것이다.
    runningText = headline + first_para + body + caption

    #Pre-processing: Tokenizer
    runningText = runningText.lower()
    tokens = nltk.word_tokenize(runningText)
    tokens = list(set(tokens))#remove duplicates

    ## Pre-processing: Stopwords(1)
    stop = set(stopwords.words('english'))
    tokens = [i for i in tokens if i not in stop]

    ## Pre-processing: Stopwords(2) :  NLTK가 Stopwords로 처리하지 못하는 것 따로 처리   
    mystop = set(punctuation)
    mystop2 = ['"',"--",'``',"''",'.',',']
    tokens = [i for i in tokens if i not in mystop]
    tokens = [i for i in tokens if i not in mystop2]

    ## Pre-processing: Stemming using SnowballStemmer    
    snowball_stemmer = SnowballStemmer("english")
    stemmedTokens= []
    for token in tokens:
        stemmedTokens.append(snowball_stemmer.stem(token)) 

    return stemmedTokens


# In[35]:

def get_stemmedTokens_fromTxt(path):
    with open(os.path.join(os.getcwd(), path)) as f:
        runningText = f.read()
        
        #Pre-processing: Tokenizer
        import nltk
        runningText = runningText.lower()
        tokens = nltk.word_tokenize(runningText)
        tokens = list(set(tokens))#remove duplicates

        ## Pre-processing: Stopwords(1)
        from nltk.corpus import stopwords 
        stop = set(stopwords.words('english'))
        tokens = [i for i in tokens if i not in stop]

        ## Pre-processing: Stopwords(2) :  NLTK가 Stopwords로 처리하지 못하는 것 따로 처리
        from string import punctuation
        mystop = set(punctuation)
        mystop2 = ['"',"--",'``',"''",'.',',']
        tokens = [i for i in tokens if i not in mystop]
        tokens = [i for i in tokens if i not in mystop2]

        ## Pre-processing: Stemming using SnowballStemmer
        from nltk.stem import SnowballStemmer
        snowball_stemmer = SnowballStemmer("english")
        stemmedTokens= []
        for token in tokens:
            stemmedTokens.append(snowball_stemmer.stem(token)) 

        return stemmedTokens


# In[36]:

globalSet=set([])
for fname in training_files: ##training file 주제별20개 즉,총60개의 파일에 대하여 stemmedToken을 얻는다.
    with open(os.path.join(os.getcwd(), fname)) as f:
        urls = f.readlines()
        for u in urls:
            text = get_stemmedTokens_fromUrl(u.strip())
            globalSet = globalSet | set(text) 
            
globalList = list(globalSet)


# # Vectorization

# In[37]:

golf_Vec = [0 for i in range(0, len(globalSet))]
ent_Vec = [0 for i in range(0, len(globalSet))]
pol_Vec = [0 for i in range(0, len(globalSet))]


# In[38]:

## golf 주제의 topicVector생성 - count scheme
golfTxt=['category/golf_20.txt']
for fname in golfTxt:
    with open(os.path.join(os.getcwd(), fname)) as f:
        urls = f.readlines()
        for u in urls:
            text = get_stemmedTokens_fromUrl(u.strip())
            for w in text:
                if w in globalList:
                    golf_Vec[globalList.index(w)] += 1  


# In[39]:

##entertainment 주제의 topicVector생성 - count scheme
entertainmentTxt=['category/entertainment_20.txt']
for fname in entertainmentTxt:
    with open(os.path.join(os.getcwd(), fname)) as f:
        urls = f.readlines()
        for u in urls:
            text = get_stemmedTokens_fromUrl(u.strip())
            for w in text:
                if w in globalList:
                    ent_Vec[globalList.index(w)] += 1


# In[40]:

##politics 주제의 topicVector생성 - count scheme             
politicsTxt=['category/politics_20.txt']
for fname in politicsTxt:
    with open(os.path.join(os.getcwd(), fname)) as f:
        urls = f.readlines()
        for u in urls:
            text = get_stemmedTokens_fromUrl(u.strip())
            for w in text:
                if w in globalList:
                    pol_Vec[globalList.index(w)] += 1


# # Classification

# In[41]:

testText = get_stemmedTokens_fromTxt(sys.argv[1])
#testText = get_stemmedTokens_fromTxt('document/politics.txt')
testText_Vec = [0 for i in range(0, len(globalSet))]

for w in testText:
    if w in globalList:
        testText_Vec[globalList.index(w)] += 1    
   
    
def cosineSimilarity(v1, v2):
        multi = (v1.dot(v2)).sum()
        x = math.sqrt((v1*v1).sum())
        y = math.sqrt((v2*v2).sum())

        result = multi/(x*y)
        return result


# In[42]:

similarity_with_golf = cosineSimilarity(np.array(golf_Vec), np.array(testText_Vec))
print('The similarity with golf is',similarity_with_golf)

similarity_with_ent = cosineSimilarity(np.array(ent_Vec), np.array(testText_Vec))
print('The similarity with entertainment is',similarity_with_ent)

similarity_with_pol = cosineSimilarity(np.array(pol_Vec), np.array(testText_Vec))
print('The similarity with politics is',similarity_with_pol,'\n')

if similarity_with_golf> similarity_with_ent  and similarity_with_golf>similarity_with_pol:
    print("This text's topic is ★golf★")
if similarity_with_ent> similarity_with_golf  and similarity_with_ent>similarity_with_pol:
    print("This text's topic is ★entertainment★")
if similarity_with_pol> similarity_with_golf  and similarity_with_pol>similarity_with_ent:
    print("This text's topic is ★politics★")

print('\n')


# # Visualization

# In[67]:

#golf
print('Golf Text Top 5 words')
x = [i for i in range (0,len(globalSet))] #x에 단어의 id(위치)를 담는다

# zip을 사용해서  x와 ent_Vec을 묶음으로써 -> id와 단어빈도수를 묶는다
topList = [(x,y) for x,y in zip(x, golf_Vec)] 
#sort를 사용해서 빈도수로 정렬(오름차순)한다. 
topList.sort(key=lambda x:x[1])  
#오름차순 정렬에서 뒤에서 5개(빈도수 최다 5개)를 뽑는다.
topList = topList[-5:]

labelOrder = [5,4,3,2,1]
#현재 topList는 (x,y)의 투플형태로 (id(위치),단어)의 데이터를 오름차순 정렬해서 가지고있다

#topList에서 id(위치)를 가져와서 globalList에 해당위치의 단어를 가져온다
topWordLabels = [globalList[x] for (x,y) in topList] 

#topList에서 단어(y)를 가져와서 globalList에 해당위치의 단어를 가져온다
topWordCounting = [y for (x,y) in topList]


plt.bar(labelOrder, topWordCounting)
plt.xticks(labelOrder, topWordLabels) 
plt.show()

#ent
print('Entertainment Text Top 5 words')
x = [i for i in range (0,len(globalSet))] #x에 단어의 id(위치)를 담는다

# zip을 사용해서  x와 ent_Vec을 묶음으로써 -> id와 단어빈도수를 묶는다
topList = [(x,y) for x,y in zip(x, ent_Vec)] 
#sort를 사용해서 빈도수로 정렬(오름차순)한다. 
topList.sort(key=lambda x:x[1])  
#오름차순 정렬에서 뒤에서 5개(빈도수 최다 5개)를 뽑는다.
topList = topList[-5:]

labelOrder = [5,4,3,2,1]
#현재 topList는 (x,y)의 투플형태로 (id(위치),단어)의 데이터를 오름차순 정렬해서 가지고있다

#topList에서 id(위치)를 가져와서 globalList에 해당위치의 단어를 가져온다
topWordLabels = [globalList[x] for (x,y) in topList] 

#topList에서 단어(y)를 가져와서 globalList에 해당위치의 단어를 가져온다
topWordCounting = [y for (x,y) in topList]


plt.bar(labelOrder, topWordCounting)
plt.xticks(labelOrder, topWordLabels) 
plt.show()

#pol
print('Entertainment Text Top 5 words')
x = [i for i in range (0,len(globalSet))] #x에 단어의 id(위치)를 담는다

# zip을 사용해서  x와 ent_Vec을 묶음으로써 -> id와 단어빈도수를 묶는다
topList = [(x,y) for x,y in zip(x, pol_Vec)] 
#sort를 사용해서 빈도수로 정렬(오름차순)한다. 
topList.sort(key=lambda x:x[1])  
#오름차순 정렬에서 뒤에서 5개(빈도수 최다 5개)를 뽑는다.
topList = topList[-5:]

labelOrder = [5,4,3,2,1]
#현재 topList는 (x,y)의 투플형태로 (id(위치),단어)의 데이터를 오름차순 정렬해서 가지고있다

#topList에서 id(위치)를 가져와서 globalList에 해당위치의 단어를 가져온다
topWordLabels = [globalList[x] for (x,y) in topList] 

#topList에서 단어(y)를 가져와서 globalList에 해당위치의 단어를 가져온다
topWordCounting = [y for (x,y) in topList]


plt.bar(labelOrder, topWordCounting)
plt.xticks(labelOrder, topWordLabels) 
plt.show()


# In[ ]:




# In[ ]:



