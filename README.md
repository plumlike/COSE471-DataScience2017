# COSE471-DataScience
COSE471 Data Science @ Korea Univ, Spring 2017

## News Topic Classifier
Classify CNN News topic

### Usage  
1. Make sure to locate the 'category' folder(for training) to where jieun.py (exe file) is located<br>
2. Run **anaconda prompt** (nltk, numpy, beautifulsoup, matplotlib must already be installed)<br>
```shell                   
$ python jieun.py golf.txt 
```
**jieun.py** is execution file and **golf.txt** is a testfile 

### Development issue
    Training data can be expanded if you add news URL to category folder.
    Using 'pos tagging' method will upgrade the accuracy of classifying articles.
