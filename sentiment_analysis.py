import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset ,DataLoader
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

df= pd.read_csv("IMDB Dataset.csv")
# print(df.head())
#print(df.shape)                                          # (50000,2)
# print(df.isnull().sum())                                # no null values
#  df.drop_duplicates(inplace= True)
# print(df)                                                 #[49582 rows x 2 columns]


df["review"].str.lower()
#print(df["review"])

#removing urls 
def remove_url(text):
    text=re.sub(r"http\s+"," ",text)
    return text

# removing punctuation
def remove_punch(text):
    text=re.sub(r"[^A-Za-z0-9\s]"," ",text)
    return text

# removing html tags
def remove_html(text):
    text=re.sub(r"<.*?>"," ",text)
    return text

df["review"]=df["review"].apply(remove_url)
df["review"]=df["review"].apply(remove_punch)
df["review"]=df["review"].apply(remove_html)


# # '''nltk'''
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")


# removing stopwords
def remove_stopwords(text):
    tokens=word_tokenize(text)
    stop_words=stopwords.words("english")

    for word in tokens:
        if word in stop_words:
            text= text.replace("word"," ")
    return text

df["review"]=df["review"].apply(remove_stopwords)
# print(df.head())

#stemmming
from nltk.stem import PorterStemmer

def stemming(text):
    ps=PorterStemmer()
    stemmed_words=word_tokenize(text)
    tokens=(text)
    for token in tokens:
        stemmed_token=ps.stem(token)
        stemmed_words.append(stemmed_token)

        return " ".join(stemmed_words)

df["review"]=df["review"].apply(stemming)

#print(df.head())


# encoding
le=LabelEncoder()
df["sentiment"]=le.fit_transform(df["sentiment"])
y=df["sentiment"]

# vecorization
tf= TfidfVectorizer(max_features=5000)
X=tf.fit_transform(df["review"])

print(type(X))

#train-test-split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train=X_train.toarray()
X_test=X_test.toarray()

#print(X_test)

trainset=TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train.values).float()
)
testset=TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test.values).float()

)

trainloader=DataLoader(trainset,shuffle=True,batch_size=64)
testloader=DataLoader(testset,batch_size=64,shuffle=True)

# building RNN module

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size=128,num_layers=1):
        super().__init__()

        self.hidden_size=hidden_size
        self.num_layers=num_layers


        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)

        self.fc=nn.Linear(hidden_size,1)


    def forward(self,x):
        h0=torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        )
        out,_=self.rnn(x,h0)
        out=self.fc(out[:,-1,:])
        return out

#model training
input_size=X_train.shape[1]
model=RNN(input_size)
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters())

# training
epochs=10

for epoch in range(epochs):
    model.train()

    for Xb,yb in trainloader:
        optimizer.zero_grad()

        Xb=Xb.unsqueeze(1)
        outputs=model(Xb).squeeze(1)
        output=torch.sigmoid(outputs)
        loss=criterion(outputs,yb)
        loss.backward()
        optimizer.step()

    print(f"epoch:  {epoch+1}/{epochs}   loss:  {loss.item()}")


# evaluation

model.eval()
with torch.no_grad():
    correct=0
    total=0
    for Xb,yb in testloader:
        Xb=Xb.unsqueeze(1)

        outputs=model(Xb)
        predicted=(torch.sigmoid(outputs.squeeze(1))>0.5).float()
        total +=yb.size(0)
        correct +=(predicted==yb).sum().item()

    print(f"accuracy={correct/total *100}")

# finished

