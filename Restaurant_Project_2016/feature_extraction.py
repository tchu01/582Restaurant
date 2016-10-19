import os, re, random, operator, nltk, collections
import review_scraper as rs
import copy
from copy import deepcopy
from itertools import chain
from nltk.corpus import stopwords as sw
from nltk.metrics.scores import recall, precision, f_measure

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
      cnt = cnt + 1
   if review['SERVICE'] is not None:
      total = total + review['SERVICE']
      cnt = cnt + 1
   if review['VENUE'] is not None:
      total = total + review['VENUE']
      cnt = cnt + 1
      
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
   average = sum([len(sent) for sent in sents]) // len(sents)      
   if average > 0 and average <= 10:
      return 'short_avg'
   elif average > 10 and average < 15:
      return 'medium_avg'
   else:
      return 'high_avg'

# Assumes paragraph is a list of paragraphs
def paragraph_features(paragraph):
   split_paragraph_by_space = copy.deepcopy(paragraph)
   split_paragraph_by_period = copy.deepcopy(paragraph)
   
   split_paragraph_by_space = split_paragraph_by_space.split()
   split_paragraph_by_period = split_paragraph_by_period.split('.')
   split_paragraph_by_period = [sentence.split() for sentence in split_paragraph_by_period]

   features = {#'lexical_diversity':lexical_diversity(split_paragraph_by_space), 
               'average_sent_length':avg_sent_length(split_paragraph_by_period)} 
   return features 
   #return {}

full = 0

def scrape1():
   subdirectories = chain(os.walk("Review1"),
                          os.walk("Review2"))
                          
   data = []
   for path in subdirectories:
      if len(path[1]) == full:
         matchName = re.match(r'(.*) (.*)', path[0])
         data.append(rs.scrape_page(path[0] + '/onlinetext.html',
                                    matchName.group(1).split('\\')[1] +
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
                                    matchName.group(1).split('\\')[1] +
                                    ' ' + matchName.group(2).split('_')[0]))

   data = [d for d in data if d]

   '''
   for d in data:
      print(d)
      print(len(data))
   '''

   return data


def bin_lex(num):
   if num <= .6:
      return 'low_lex'
   if num <= .7:
      return 'mid_low_lex'
   if num >= .9:
      return 'high_lex'
   if num >= .8:
      return 'mid_high_lex'

def group_pars(review):
   return '  '.join(review)


def binary(item):
   if item < 4:
      return 0
   else:
      return 1

def cat_bin(review):
   mapping = {1/3 : "a", 2/3 : "b", 3/3 : "c", 1/2 : "d", 0 : "e"}
   cnt = 0
   total = 0
   if review['FOOD'] is not None:
      total = total + review['FOOD']
      cnt = cnt + 1
   if review['SERVICE'] is not None:
      total = total + review['SERVICE']
      cnt = cnt + 1
   if review['VENUE'] is not None:
      total = total + review['VENUE']
      cnt = cnt + 1
      
   if cnt != 0:
      return mapping[total/cnt]
   else:
      return -1

def avg_par_len(review):
   split_pars_lens = [len(par.split()) for par in review['review'].copy()]
   avg_len = sum(split_pars_lens) / len(split_pars_lens)
   if avg_len < 60:
      return 'low'
   if avg_len < 100:
      return 'mid'
   if avg_len > 99:
      return 'high'

def make_bigrams(review):
   bigram_pars = [nltk.bigrams(par.split()) for par in review['review'].copy()]
   bigrams = []
   for bigram_par in bigram_pars:
      for bigram in bigram_par:
         bigrams.append(bigram)
   return bigrams


def make_corpus(reviews):
   bigrams = []
   for review in reviews:
      for bigram in make_bigrams(review):
         bigrams.append(bigram)
   return bigrams

#get top 30
#run through 

def append_bigrams(ap_dict, rev_com, corp_com):
   it = 0
   corp = [i[0] for i in corp_com]
   #print(corp)
   dict_cpy = dict(ap_dict.copy())
   for bigram in corp:
      #print(bigram)
      dict_cpy['bigram_' + str(it)] = bigram in corp
      if bigram[0] in corp:
         print("TRUE")
      it += 1
   return dict_cpy

def predict_author(train_scrape, test_scrape):
   #use lexical diversity and average sentence length
   corpus_dist_commons = nltk.FreqDist(make_corpus(train_scrape)).most_common()[100:130]
   train_data = []
   test_data = []
   #train
   for rev in train_scrape:
      rev_dict = {}
      rev_bigrams_common = nltk.FreqDist(make_bigrams(rev)).most_common()[0:30]
      pars = []
      reviewer = ""
      if rev['REVIEWER']:
         reviewer = rev['REVIEWER']
      if rev['review']:
         pars = group_pars(rev['review'])
      rev_dict['lex_dev'] = bin_lex(lexical_diversity(pars.split()))
      rev_dict['avg_par_len'] = avg_par_len(rev)
      rev_dict = append_bigrams(rev_dict, rev_bigrams_common, corpus_dist_commons)
      train_data.append((rev_dict, reviewer))

   #test
   for rev in test_scrape:
      rev_dict = {}
      rev_bigrams_common = nltk.FreqDist(make_bigrams(rev)).most_common()[0:30]
      pars = []
      reviewer = ""
      if rev['REVIEWER']:
         reviewer = rev['REVIEWER']
      if rev['review']:
         pars = group_pars(rev['review'])
      rev_dict['lex_dev'] = bin_lex(lexical_diversity(pars.split()))
      rev_dict['avg_par_len'] = avg_par_len(rev)
      rev_dict = append_bigrams(rev_dict, rev_bigrams_common, corpus_dist_commons)
      #print(rev_dict)
      test_data.append((rev_dict, reviewer))

   classifier = nltk.NaiveBayesClassifier.train(train_data)
   refsets = collections.defaultdict(set)
   testsets = collections.defaultdict(set)
   for i, (feats, label) in enumerate(testsets):
      refsets[label].add(i)
      observed = classifier.classify(feats)
      testsets[observed].add(i)
   
   print("Accuracy: ",nltk.classify.accuracy(classifier,test_data))
   print(classifier.show_most_informative_features(20))   


#prob 

#normalize based on how much they wrote/number of tokens
#you hve list of classes and associated bigrams
#you also have everybodies bigrams, get a probdist
#pull out bigrams
#for each class i've seen what is the rpob 
#that it is associated with theis class and in general what is hte prob of this class 
#there is

#get bigrams
#words
#three diff

tr = True


if __name__ == '__main__':
   if tr:
      predict_author(scrape1(), scrape2())
         
   # Testing helper functions
   '''
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

   # List of reviews, which are dictionaries
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

   good_ratings = 0
   bad_ratings = 0
   for (feature,rating) in train_data:
      if rating == 1:
         good_ratings = good_ratings + 1
      if rating == 0:
         bad_ratings = bad_ratings + 1
   print("Training... Good ratings: " + str(good_ratings) + " | Bad ratings: " + str(bad_ratings))
   print("Good/Total: " + str(good_ratings/(good_ratings+bad_ratings)))

   good_ratings = 0
   bad_ratings = 0
   for (feature,rating) in test_data:
      if rating == 1:
         good_ratings = good_ratings + 1
      if rating == 0:
         bad_ratings = bad_ratings + 1
   print("Testing... Good ratings: " + str(good_ratings) + " | Bad ratings: " + str(bad_ratings))
   print("Good/Total: " + str(good_ratings/(good_ratings+bad_ratings)))

   #print(train_data[:5])
   #print(test_data[:5])
   
   classifier = nltk.NaiveBayesClassifier.train(train_data)
   print("Accuracy: ",nltk.classify.accuracy(classifier,test_data))
   print(classifier.show_most_informative_features(20))
   '''