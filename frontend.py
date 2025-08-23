import streamlit as st
import json
import re
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

st.title("Simple Message Spam Filter using Naive Bayes Classifier")
with st.form("user_input"):
    message=st.text_area(label="Enter your message",height=150,label_visibility="hidden",placeholder="Enter your message")
    submitted=st.form_submit_button(label="Send")
    
if submitted:
    if classify(message) == 1:
        st.subheader("👀SPAM!!!")
    else:
        st.subheader("NOT SPAM")