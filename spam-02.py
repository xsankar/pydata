#!/usr/bin/env python
#
# pydata Tutorial March 18, 2013
# Thanks to StreamHacker a.k.a. Jacob Perkins
#
import os
import sys
import time
import collections
#
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk import tokenize
from nltk import metrics

#
# Take a list of words and turn it into a Bag Of Words
#
def bag_of_words(words):
    return dict([(word, True) for word in words])

start_time = time.clock()

#
# Data
#
ham_list=[]
spam_list=[]
#
root_path = '/Users/ksankar/Documents/Erlang/Bayesian/spam/enron-spam/enron-pre-processed/'
directories = ['enron1','enron2','enron3','enron4','enron5','enron6']
#directories = ['enron5'] #['enron1','enron2','enron3','enron4','enron5','enron6']
#directories = ['enron-small'] # ['enron5'] #['enron1','enron2','enron3','enron4','enron5','enron6']
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
# print ham_features[0]
# print spam_features[0]
# print ham_features[1]
# print spam_features[1]

trainfeats = ham_features[:ham_cutoff] + spam_features[:spam_cutoff]
testfeats = ham_features[ham_cutoff:] + spam_features[spam_cutoff:]
print 'Train on %d instances, Test on %d instances' % (len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
#
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
#print len(refsets)
#print len(testsets)
#print refsets
precisions = {}
recalls = {}
for label in classifier.labels():
    precisions[label] = metrics.precision(refsets[label],testsets[label])
    recalls[label] = metrics.recall(refsets[label], testsets[label])
c_00=len(refsets[labels[0]].intersection(testsets[labels[0]]))
c_01=len(refsets[labels[0]].intersection(testsets[labels[1]]))
c_10=len(refsets[labels[1]].intersection(testsets[labels[0]]))
c_11=len(refsets[labels[1]].intersection(testsets[labels[1]]))
print '  |   H   |   S   |'
print '--|-------|-------|'
print 'H | %5d | %5d |' % (c_00,c_01)
print '--|-------|-------|'
print 'S | %5d | %5d |' % (c_10,c_11)
print '--|-------|-------|'

ham_p = float(c_00) / (c_00 + c_10)
ham_r = float(c_00) / (c_00 + c_01)
spam_p = float(c_11) / (c_01 + c_11)
spam_r = float(c_11) / (c_10 + c_11)

print 'P:{ham:%f,spam:%f}' % (ham_p,spam_p)
print 'R:{ham:%f,spam:%f}' % (ham_r,spam_r)

print '(nltk)P:',precisions
print '(nltk)R:',recalls

""" Sample Output
Run 1:

enron5 ham Total Files = 1500 Feature Count = 1500
enron5 spam Total Files = 5175 Feature Count = 3675
Read Time : 80.081
train on 4657 instances, test on 518 instances
P: {'ham': 0.9803921568627451, 'spam': 1.0}
R: {'ham': 1.0, 'spam': 0.9918478260869565}
  |   H   |   S   |
--|-------|-------|
H |   150 |     0 |
--|-------|-------|
S |     3 |   365 |
--|-------|-------|
P:{ham:0.980392,spam:1.000000}
R:{ham:1.000000,spam:0.991848}
Run Time : 100.082
That's all Folks! All Good Things have to come to an end !
--------------------------------------
Run 2:
enron1 ham Total Files = 3672 Feature Count = 3672
enron1 spam Total Files = 5172 Feature Count = 1500
enron2 ham Total Files = 9533 Feature Count = 8033
enron2 spam Total Files = 11029 Feature Count = 2996
enron3 ham Total Files = 15041 Feature Count = 12045
enron3 spam Total Files = 16541 Feature Count = 4496
enron4 ham Total Files = 18041 Feature Count = 13545
enron4 spam Total Files = 22541 Feature Count = 8996
enron5 ham Total Files = 24041 Feature Count = 15045
enron5 spam Total Files = 27716 Feature Count = 12671
enron6 ham Total Files = 29216 Feature Count = 16545
enron6 spam Total Files = 33716 Feature Count = 17171
Read Time : 581.212
train on 30343 instances, test on 3373 instances
  |   H   |   S   |
--|-------|-------|
H |  1630 |    25 |
--|-------|-------|
S |    38 |  1680 |
--|-------|-------|
P:{ham:0.977218,spam:0.985337}
R:{ham:0.984894,spam:0.977881}
(nltk)P: {'ham': 0.9772182254196643, 'spam': 0.9853372434017595}
(nltk)R: {'ham': 0.9848942598187311, 'spam': 0.9778812572759022}
Run Time : 703.728
That's all Folks! All Good Things have to come to an end !

"""

end_time = time.clock()
print "Run Time : %1.3f" % (end_time-start_time)
print "That's all Folks! All Good Things have to come to an end !"
#sys.exit()
#
# That's all Folks!
#
