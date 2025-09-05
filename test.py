import pandas as pd
import re
import json
import math

#Fetching Model Metadata
with open("model_metadata.json","r") as f:
    model_metadata=json.load(f)

#Fetching Ham Words Dictionary
with open("ham_words_dict.json","r") as f:
    ham_words_dict=json.load(f)

#Fetching Spam Words Dictionary
with open("spam_words_dict.json","r") as f:
    spam_words_dict=json.load(f)

# Tokenizer function to break the message into tokens
def tokenizer(message):
    return re.findall(r'\b\w+\b',message.lower())

# Classification function to classify the message as spam or ham
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
total_actual_positives=0
total_pred_positives=0
true_positives=0
true_negatives=0
false_positives=0
false_negatives=0

for label in test_data["label"]:
    if label == 1:
        total_actual_positives+=1

for label,message in zip(test_data["label"],test_data["message"]):
    prediction=classify(message)
    if prediction == label:
        correct_predictions+=1
    if prediction == 0:
        if label == 0:
            true_negatives+=1
        elif label == 1:
            false_negatives+=1
    if prediction == 1:
        total_pred_positives+=1
        if label == 1:
            true_positives+=1
        elif label == 0:
            false_positives+=1

accuracy = (correct_predictions/len(test_data)) * 100
precision = true_positives/total_pred_positives if total_pred_positives != 0 else 0
recall = true_positives/total_actual_positives if total_actual_positives != 0 else 0
print("Metrics(in percentage): ")
print("Accuracy of the Naive Bayes Classifier on unseen data = ",accuracy)
print("Precision = ",precision*100)
print("Recall = ",recall*100)
print("Confusion Matrix :")
print(f"{'':<15}{'Pred_Spam':<15}Pred_Ham")
print(f"{'Actual Spam':<15}{true_positives:<15}{false_negatives}")
print(f"{'Actual Ham':<15}{false_positives:<15}{true_negatives}")