import pandas as pd
import re
import json
import math

with open("model_metadata.json","r") as f:
    model_metadata=json.load(f)

with open("ham_words_dict.json","r") as f:
    ham_words_dict=json.load(f)

with open("spam_words_dict.json","r") as f:
    spam_words_dict=json.load(f)

def tokenizer(message):
    return re.findall(r'\b\w+\b',message.lower())

def classify(message):
    words=tokenizer(message)
    log_prob_spam = model_metadata["log_of_probability_of_spam"]
    log_prob_ham = model_metadata["log_of_probability_of_ham"]
    for word in words:
        log_prob_spam += math.log((spam_words_dict.get(word,0)+1)/(model_metadata["total_spam_words"]+model_metadata["vocab_size"]))
        log_prob_ham += math.log((ham_words_dict.get(word,0)+1)/(model_metadata["total_ham_words"]+model_metadata["vocab_size"]))
    if log_prob_ham > log_prob_spam:
        return 0
    elif log_prob_spam > log_prob_ham:
        return 1
    else:
        return 0
    
test_data=pd.read_csv("test.csv",encoding="latin-1",usecols=[0,1])

correct_predictions=0

for label,message in zip(test_data["label"],test_data["message"]):
    if classify(message) == label:
        correct_predictions+=1

accuracy = (correct_predictions/len(test_data)) * 100
print("Accuracy of the Naive Bayes Classifier on unseen data = ",accuracy)