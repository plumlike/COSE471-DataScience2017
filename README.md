# COSE471-DataScience
COSE471 Data Science @ Korea Univ, Spring 2017

## News Topic Classifier
Classify CNN News topic

### Usage  
1. Make sure the'category' folder(for training) and python script file(jieun.py) be in the same directory<br>
2. Run **anaconda prompt** (nltk, numpy, beautifulsoup, matplotlib must be installed)<br>
```shell                   
$ python jieun.py golf.txt 
```
**jieun.py** is execution file and **golf.txt** is a testfile 

### Development issue
    Training data can be expanded if you add news URL to category folder.
    Using 'pos tagging' method will upgrade the accuracy of classifying articles.
