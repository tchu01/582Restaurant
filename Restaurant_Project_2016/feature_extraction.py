import os, re, random, operator, nltk
import review_scraper as rs
import copy
from copy import deepcopy
from itertools import chain
from nltk.corpus import stopwords as sw

# Removes stopwords from a paragraph
def remove_stopwords(paragraph):
   short = paragraph.split()
   short = [word for word in short if word not in sw.words("english")]
   return ' '.join(short) 

# Finds the average score of food, service, venue for each review
def average_total_score(review):
   total = 0
   cnt = 0;
   if review['FOOD'] is not None:
      total = total + review['FOOD']
      cnt = count + 1
   if review['SERVICE'] is not None:
      total = total + review['SERVICE']
      cnt = count + 1
   if review['VENUE'] is not None:
      total = total + review['VENUE']
      cnt = count + 1
      
   if cnt != 0:
      return total/cnt
   else:
      return -1

# Creates a dictionary for all reviewers for one review number (ie: Review#1)
# and calculates their avg score and lists their overall score.
def review_scores_per_person(dictionary):
   scores_per_person = {}
   for key in dictionary:
      scores_per_person[key] = average_total_score(dictionary[key])
      
   return scores_per_person

# Creates a list of the n most common words in the text
# Can be used to extract common words from all review with score 4-5 or 1-3
def common_words(text, n):
   fdist = nltk.FreqDist(text)
   return [word for word,count in fdist.most_common(n)]

# Assumes text passed in as list of words
def lexical_diversity(text):
   return len(set(text)) / len(text)

good_keywords = ['good','great','excellent','fantastic', 'love', 'like']
bad_keywords = ['bad','terrible','horrible','unsatisfactory', 'hate', 'dislike']

# Assumes text passed in as list of words
def count_good_keywords(text):
   i = 0
   for words in text:
      if words in good_keywords:
         i = i + 1

   return i

# Assumes text passed in as list of words
def count_bad_keywords(text):
   i = 0
   for words in text:
      if words in bad_keywords:
         i = i + 1

   return i

# Assumes sents passed in as list of sentences, which is assumed to be list of words.
def avg_sent_length(sents):
   return sum([len(sent) for sent in sents]) // len(sents)      

# Assumes paragraph is a list of paragraphs
def paragraph_features(paragraph):
   split_paragraph_by_space = copy.deepcopy(paragraph)
   split_paragraph_by_period = copy.deepcopy(paragraph)
   
   split_paragraph_by_space = split_paragraph_by_space.split()
   split_paragraph_by_period = split_paragraph_by_period.split('.')
   split_paragraph_by_period = [sentence.split() for sentence in split_paragraph_by_period]

   features = {'lexical_diversity':lexical_diversity(split_paragraph_by_space), 
               'average_sent_length':avg_sent_length(split_paragraph_by_period)} 
   return features 

full = 0

def scrape1():
   subdirectories = chain(os.walk("Review1"),
                          os.walk("Review2"),
                          os.walk("Review3"))

   data = []
   for path in subdirectories:
      if len(path[1]) == full:
         matchName = re.match(r'(.*) (.*)', path[0])
         data.append(rs.scrape_page(path[0] + '/onlinetext.html',
                                    matchName.group(1).split('/')[1] +
                                    ' ' + matchName.group(2).split('_')[0]))

   data = [d for d in data if d]

   '''
   for d in data:
      print(d)
      print(len(data))
   '''

   return data

def scrape2():
   subdirectories = os.walk("Review3")

   data = []
   for path in subdirectories:
      if len(path[1]) == full:
         matchName = re.match(r'(.*) (.*)', path[0])
         data.append(rs.scrape_page(path[0] + '/onlinetext.html',
                                    matchName.group(1).split('/')[1] +
                                    ' ' + matchName.group(2).split('_')[0]))

   data = [d for d in data if d]

   '''
   for d in data:
      print(d)
      print(len(data))
   '''

   return data

if __name__ == '__main__':
   # Testing helper functions
   text = ["hello", "my", "name", "is", "Tim", "Tim", "hello", "good", "bad", "horrible"]
   common = common_words(text, 2)
   print("Text: " + str(text))
   print("Common words: " + str(common))
   print("Lexical Diversity: " + str(lexical_diversity(text)))
   print("Count good words: " + str(count_good_keywords(text)))
   print("Count bad words: " + str(count_bad_keywords(text)))

   print()
   sents = [["hello", "tim"], ["bye", "tim", "chu"]]
   print("Sents: " + str(sents))
   print("Avg sent length: " + str(avg_sent_length(sents)))

   print()
   print(remove_stopwords("Hello this is my sentence"))

   train = scrape1()
   test = scrape2()

   train_data = []
   for review in train:
      #print()
      #print("Reviewer: " + str(review['REVIEWER']))
      cnt = 0
      for paragraph in review['review']:
         if len(paragraph) > 15:
            #print("Paragraph: " + str(paragraph))
            if cnt == 0:
               train_tuple = (paragraph_features(paragraph), review['FOOD'])
               train_data.append(train_tuple)
               cnt = cnt + 1
            elif cnt == 1:
               train_tuple = (paragraph_features(paragraph), review['SERVICE'])
               train_data.append(train_tuple)
               cnt = cnt + 1
            elif cnt == 2:
               train_tuple = (paragraph_features(paragraph), review['VENUE'])
               train_data.append(train_tuple)
               cnt = cnt + 1
            elif cnt == 3:
               train_tuple = (paragraph_features(paragraph), review['OVERALL'])
               train_data.append(train_tuple)
               cnt = cnt + 1

   test_data = []
   for review in test:
      #print()
      #print("Reviewer: " + str(review['REVIEWER']))
      cnt = 0
      for paragraph in review['review']:
         if len(paragraph) > 15:
            #print("Paragraph: " + str(paragraph))
            if cnt == 0:
               test_tuple = (paragraph_features(paragraph), review['FOOD'])
               test_data.append(test_tuple)
               cnt = cnt + 1
            elif cnt == 1:
               test_tuple = (paragraph_features(paragraph), review['SERVICE'])
               test_data.append(test_tuple)
               cnt = cnt + 1
            elif cnt == 2:
               test_tuple = (paragraph_features(paragraph), review['VENUE'])
               test_data.append(test_tuple)
               cnt = cnt + 1
            elif cnt == 3:
               test_tuple = (paragraph_features(paragraph), review['OVERALL'])
               test_data.append(test_tuple)
               cnt = cnt + 1

   print(train_data[:5])
   print(test_data[:5])
   
   classifier = nltk.NaiveBayesClassifier.train(train_data)
   print("Accuracy: ",nltk.classify.accuracy(classifier,test_data))
   print(classifier.show_most_informative_features(20))
