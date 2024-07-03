import numpy as np
import pandas as pd
import pickle
import re
import string

from nltk.stem import PorterStemmer
ps = PorterStemmer()


#load model
with open("static/model/model.pickle","rb") as f:
    model = pickle.load(f)

#load stopwords
with open("static/model/corpora/stopwords/english",'r') as file:
          sw= file.read().splitlines()

#load tokens
vocab = pd.read_csv("static/model/vocabulary.txt",header= None)
tokens = vocab[0].tolist()

def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation,' ')
    return text

def preprocessing(text):
    data = pd.DataFrame([text],columns=["tweet"])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))  #upperto lower
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(re.sub(r'https?:\/\/.*[\r\n]*', ' ', x, flags =re.MULTILINE) for x in x.split())) #remove links
    data["tweet"] =data["tweet"].apply(remove_punctuation) #remove punctuation
    data["tweet"]=data["tweet"].str.replace('\d+','',regex = True) #rmov numbers
    data ["tweet"]=data ["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))  #remove stopwords
    data["tweet"] = data ["tweet"].apply(lambda x: ' '.join(ps.stem(x) for x in x.split()))
    return data["tweet"]

def vectorization(ds):
    vectorized_lst=[]
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] =1
        vectorized_lst.append(sentence_lst)
    vectorized_lst_new = np.asarray(vectorized_lst,dtype = np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction ==1:
        return "Negative"
    else:
        return "Positive"
