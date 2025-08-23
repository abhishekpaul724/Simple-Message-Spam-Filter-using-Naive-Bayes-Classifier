import streamlit as st

st.title("Simple Message Spam Filter using Naive Bayes Classifier")
with st.form("user_input"):
    message=st.text_area(label="Enter your message",height=150,label_visibility="hidden",placeholder="Enter your message")
    submitted=st.form_submit_button(label="Send")
    
if submitted:
    st.subheader("SPAM!!!")