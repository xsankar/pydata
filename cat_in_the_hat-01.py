#!/usr/bin/env python
#
# pydata Tutorial March 18, 2013
# Thanks to StreamHacker a.k.a. Jacob Perkins
# Thanks to Prof.Todd Ebert
#
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk import tokenize
#
# Data
#
label_1 = "Cat In The Hat"

train_text_1 = "So we sat in the house all that cold, wet day. \
And we saw him! The Cat in the Hat! \
Your mother will not mind. \
He should not be here when your mother is out. \
With a cake on the top of my hat! \
I can hold the ship."

test_text_1 = "That cat is a bad one,\
That Cat in the Hat. \
He plays lots of bad tricks. \
Don't you let him come near."

label_2 = "Green Eggs and Ham"
train_text_2 = "I am Sam. \
Sam I am.\
I do not like green eggs and ham. \
I would not like them anywhere. \
Would you like them in a house? \
I will not eat them on a train. \
Thank you, thank you, Sam I am."

test_text_2 = "I would not eat them\
here or there.\
I would not eat them anywhere.\
I would not eat green eggs and ham."

#
# For testing classification
#
classify_cith = "We saw the cat in the house."
classify_geah = "And I will eat them in a house."
classify_other = "A man a plan a canal Panama!"
#
# Take a list of words and turn it into a Bag Of Words
#
def bag_of_words(words):
    return dict([(word, True) for word in words])
#
# Program Starts Here
#
#
# Step 1 : Feature Extraction
#
train_words_1 = tokenize.word_tokenize(train_text_1)
train_words_2 = tokenize.word_tokenize(train_text_2)
train_cith = [(bag_of_words(train_words_1), label_1)]
train_geah = [(bag_of_words(train_words_2), label_2)]

test_words_1 = tokenize.word_tokenize(test_text_1)
test_words_2 = tokenize.word_tokenize(test_text_2)
test_cith = [(bag_of_words(test_words_1), label_1)]
test_geah = [(bag_of_words(test_words_2), label_2)]

train_features = train_cith + train_geah
test_features = test_cith + test_geah

#
# Step 2: Train The Dragon er ... classifier ;o)
#
classifier = NaiveBayesClassifier.train(train_features)
print 'Accuracy : %d' % nltk.classify.util.accuracy(classifier, test_features)
#print classifier.most_informative_features()
#print classifier.labels()
#
# Step 3 :Test Classification
#
print classifier.classify(bag_of_words(tokenize.word_tokenize(classify_cith)))
print classifier.classify(bag_of_words(tokenize.word_tokenize(classify_geah)))
print classifier.classify(bag_of_words(tokenize.word_tokenize(classify_other)))
#
# That's all Folks!
#
