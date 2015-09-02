import csv
import re
from nltk.corpus import stopwords
import nltk
import sys


def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

def processTweet(tweet):
	#converting to lower case
	tweet = tweet.lower()
    #removing www. and https://
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #removing username
	tweet = re.sub('@[^\s]+','',tweet)
    #removing additional white spaces
	tweet = re.sub('[\s]+', ' ', tweet)
    #removing hashtags
	tweet = re.sub(r'#([^\s]+)','', tweet)
    #trim
	tweet = tweet.strip('\'"')
	return tweet

def getFeatureVector(tweet):
	stopW = stopwords.words('english')
	featureVector = []
    #split tweet into words
	words = tweet.split()
	for w in words:
        #replace two or more with two occurrences
		w = replaceTwoOrMore(w)
        #strip punctuation
		w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
		val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
		if(w in stopW or val is None):
			continue
		else:
			featureVector.append(w.lower())
	return featureVector

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

def train(featureList, tweets):
	with open('testdata.csv','rb') as csvRead:
		for row in csvRead:
			splitted = row.split(',')
			#The data set contains the polarity ( 0 for negative, 4 for positive) in the first row
			#And the text of the tweet in the 6th row
			if(splitted[0] == '0'):
				tweet = processTweet(splitted[5])
				FV = getFeatureVector(tweet)
				sentiment = "negative"
				featureList.extend(FV)
				tweets.append((FV, sentiment));
			elif splitted[0] == '4':
				tweet = processTweet(splitted[5])
				FV = getFeatureVector(tweet)
				featureList.extend(FV)
				sentiment = "positive"
				tweets.append((FV, sentiment));
	featureList = list(set(featureList))
	#printing the featureList
	training_set = nltk.classify.util.apply_features(extract_features, tweets)
	#training the classifier
	NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
	return NBClassifier


featureList = []
tweets = []

testTweet = sys.argv[1]
if len(testTweet) < 1:
	print "Kindly enter the tweet within double quotes."
else:
	print testTweet
	NBClassifier = train(featureList, tweets)
	processedTestTweet = processTweet(testTweet)
	print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
