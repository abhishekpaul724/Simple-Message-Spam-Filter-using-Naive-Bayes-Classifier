import pandas as pd
from collections import defaultdict
import re
import json
import math

spam=pd.read_csv("spam.csv",usecols=[0,1],names=["label","message"],encoding="latin-1",skiprows=1)
spam["label"]=spam["label"].map({"ham":0,"spam":1})
spam.to_csv("refined_data.csv",index=False)

refined_data=pd.read_csv("refined_data.csv",usecols=[0,1],encoding="latin-1")
refined_data=refined_data.sample(frac=1,random_state=42)
split=int(len(refined_data)*0.8)
train=refined_data[:split]
test=refined_data[split:]
train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)

train_data=pd.read_csv("train.csv",usecols=[0,1],encoding="latin-1")

total_spam=(train_data["label"] == 1).sum()
total_ham=(train_data["label"] == 0).sum()
total_messages=len(train_data)

ham_words_dict=defaultdict(int)
spam_words_dict=defaultdict(int)
total_ham_words=0
total_spam_words=0

def tokenizer(message):
    return re.findall(r'\b\w+\b',message.lower())

for label,message in zip(train_data["label"],train_data["message"]):
    words=tokenizer(message)
    for word in words:
        if label == 1:
            spam_words_dict[word]+=1
            total_spam_words+=1
        if label == 0:
            ham_words_dict[word]+=1
            total_ham_words+=1

vocabulary = set(spam_words_dict.keys()) | set(ham_words_dict.keys())
vocab_size=len(vocabulary)

with open("spam_words_dict.json","w") as f:
    json.dump(spam_words_dict,f,indent=4)

with open("ham_words_dict.json","w") as f:
    json.dump(ham_words_dict,f,indent=4)

model_metadata={
    "total_spam": int(total_spam),
    "total_ham": int(total_ham),
    "total_messages": int(total_messages),
    "probability_of_spam": float(total_spam/total_messages),
    "probability_of_ham": float(total_ham/total_messages),
    "log_of_probability_of_spam": float(math.log(total_spam/total_messages)),
    "log_of_probability_of_ham": float(math.log(total_ham/total_messages)),
    "total_spam_words": int(total_spam_words),
    "total_ham_words": int(total_ham_words),
    "vocab_size": int(vocab_size)
}

with open("model_metadata.json","w") as f:
    json.dump(model_metadata,f,indent=4)