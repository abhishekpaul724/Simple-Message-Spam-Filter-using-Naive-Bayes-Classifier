Dataset Taken From: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


No. of spam Messages in train.csv =  590

No. of ham messages in train.csv =  3867

Total no. of messsages in train.csv=  4457


Prior Probabilities:

P(Spam) = 590/4457 = 0.13237603769351583

P(Ham) = 3867/4457 = 0.8676239623064842


After Tokenization:

total spam words =  17156

total ham words =  58512

vocab size = 7679


Formulae:

P(word|spam)=(freq. of word in spam + 1)/(total spam words + vocab size)

P(word|ham)=(freq. of word in ham + 1)/(total ham words + vocab size)


We must use logarithms to prevent numerical underflow.