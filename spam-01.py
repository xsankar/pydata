#!/usr/bin/env python
#
# pydata Tutorial March 18, 2013
# Thanks to StreamHacker a.k.a. Jacob Perkins
# Thanks to Metsis, Androutsopoulos & Paliouras
#
import os
import sys
import time
#
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk import tokenize

#
# Take a list of words and turn it into a Bag Of Words
#
def bag_of_words(words):
    return dict([(word, True) for word in words])

start_time = time.clock()

#
# Data
#
label_1 = "ham"
label_2 = "spam"
ham_list=[]
spam_list=[]
#
root_path = '/Users/ksankar/Documents/Erlang/Bayesian/spam/enron-spam/enron-pre-processed/'
directories = ['enron5'] #['enron1','enron2','enron3','enron4','enron5','enron6']
labels=['ham','spam']
#
# Need separate lists so that we can vary the proportions
#
ham_features=[]
spam_features=[]
tot_files=0
for directory in directories:
    label = labels[0] # ham
    dir_name = os.path.join(root_path,directory,label)
    #print dir_name
    for file in os.listdir(dir_name):
        file_name_full = os.path.join(dir_name,file)
        #print 'Reading file %s' % (file_name_full)
        #print '.',
        in_buffer = open(file_name_full,'r').read()
        ham_features.append((bag_of_words(tokenize.word_tokenize(in_buffer)), label)) # ham
        tot_files += 1
    print "%s %s Total Files = %d Feature Count = %d" % (directory, label, tot_files, len(ham_features))
    #
    label = labels[1] # spam
    dir_name = os.path.join(root_path,directory,label)
    for file in os.listdir(dir_name):
        file_name_full = os.path.join(dir_name,file)
        in_buffer = open(file_name_full,'r').read()
        spam_features.append((bag_of_words(tokenize.word_tokenize(in_buffer)), label)) # spam
        tot_files += 1
    print "%s %s Total Files = %d Feature Count = %d" % (directory, label, tot_files, len(spam_features))
read_time = time.clock()
print "Read Time : %1.3f" % (read_time-start_time)

#
# Now ceate the training and test data sets
#
ham_cutoff=len(ham_features) * 9/10 #3/4
spam_cutoff=len(spam_features) * 9/10 # 3/4
#
print ham_features[0]
print spam_features[0]
print ham_features[1]
print spam_features[1]

trainfeats = ham_features[:ham_cutoff] + spam_features[:spam_cutoff]
testfeats = ham_features[ham_cutoff:] + spam_features[spam_cutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print classifier.labels()
print 'Accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()

#
end_time = time.clock()
print "Run Time : %1.3f" % (end_time-start_time)
print "That's all Folks! All Good Things have to come to an end !"
#sys.exit()
#
# That's all Folks!
#
