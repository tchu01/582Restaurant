import os, re, random, operator, nltk, collections
import review_scraper as rs
import copy
import math
from copy import deepcopy
from itertools import chain
from nltk import pos_tag
from nltk.text import TextCollection
from nltk.metrics.scores import precision, recall, f_measure
from nltk.corpus import stopwords as sw
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import movie_reviews as mr

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
def count_good_words(text, good_words):
   i = 0
   for words in text:
      if words in good_words:
         i = i + 1

   return i

# Assumes text passed in as list of words
def count_bad_words(text, bad_words):
   i = 0
   for words in text:
      if words in bad_words:
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

def sentiment_analysis(paragraph):
   pos = 0
   neg = 0
   for word in paragraph:
      senti_synsets = list(swn.senti_synsets(word, "a"))
      if len(senti_synsets) > 0:
         #print(senti_synsets[0])
         pos = pos + senti_synsets[0].pos_score()
         neg = neg + senti_synsets[0].neg_score()

   #print("Pos: " + str(pos) + " NumPos: " + str(numpos))
   #print("Neg: " + str(neg) + " NumNeg: " + str(numneg))
   
   if pos > neg:
      return 'positive_sentiment'
   else:
      return 'negative_sentiment'

'''
sw_list = sw.words("english")
good_words_movies = mr.words(categories=['pos'])
good_words_movies = nltk.FreqDist(word for word in good_words_movies if word.lower() not in sw_list and word.isalpha() == True).most_common(50)
bad_words_movies = mr.words(categories=['neg'])
bad_words_movies = nltk.FreqDist(word for word in bad_words_movies if word.lower() not in sw_list and word.isalpha() == True).most_common(50)
#print(good_words_movies)
#print(bad_words_movies)
only_good_words_movies = list(set([word for word,count in good_words_movies]) - set([word for word,count in bad_words_movies]))
only_bad_words_movies = list(set([word for word,count in bad_words_movies]) - set([word for word,count in good_words_movies]))
#print(only_good_words_movies)
#print(only_bad_words_movies)
'''

# Assumes paragraph is a list of paragraphs
def paragraph_features(paragraph, good_words, bad_words):
   split_paragraph_by_space = copy.deepcopy(paragraph)
   split_paragraph_by_space = split_paragraph_by_space.split()
   
   good_count = count_good_words(split_paragraph_by_space, good_words)
   bad_count = count_bad_words(split_paragraph_by_space, bad_words)
   result = 'more_good_words'
   if good_count - bad_count < 0:
      result = 'more_bad_words'

   '''
   good_count_movies = count_good_words(split_paragraph_by_space, only_good_words_movies)
   bad_count_movies = count_bad_words(split_paragraph_by_space, only_bad_words_movies)
   result_movies = 'more_good_words'
   if good_count_movies - bad_count_movies < 0:
      result_movies = 'more_bad_words'
   '''
      
   '''
   reduced_paragraph = [word for word in split_paragraph_by_space if word.lower() not in sw.words('english')
                                                                  and word.lower() != 'food' and word.lower() != 'service'
                                                                  and word.lower() != 'venue' and word.lower() != 'restaurant'] 
   reduced_paragraph = [nltk.pos_tag([word])[0] for word in reduced_paragraph]
   reduced_paragraph = [word for (word, pos) in reduced_paragraph if pos == 'JJ']
   #print(reduced_paragraph)
   
   sentiment = sentiment_analysis(reduced_paragraph)
   '''

   features = {#'lexical_diversity':lexical_diversity(split_paragraph_by_space), 
               #'average_sent_length':avg_sent_length(split_paragraph_by_period)
               #'word_counts_movies': result_movies,
               #'sentiment': sentiment,
               'word_counts': result}
               
   return features 
   #return {}

def review_features(review, good_words, bad_words, classifier):
   average = 0;
   count = 0;
   if review['FOOD_RATING'] is not None:
      average += review['FOOD_RATING']
      count += 1
   if review['SERVICE_RATING'] is not None:
      average += review['SERVICE_RATING']
      count += 1
   if review['VENUE_RATING'] is not None:
      average += review['VENUE_RATING']
      count += 1
   
   if count != 0:
      average = math.ceil(average/count)
   else:
      average = 'no_avg'

   num_paragraphs = 0
   paragraphs = []
   for paragraph in review['review']:
      if len(paragraph) > 15:
         paragraphs.append(paragraph)      
         num_paragraphs += 1
      if num_paragraphs == 4:
         break
   
   counter = {'1': 0, '0': 0}
   num_ones = 0
   num_zeroes = 0
   for paragraph in paragraphs:
      if classifier.classify(paragraph_features(paragraph, good_words, bad_words)) == 1:
         num_ones += 1
      else:
         num_zeroes += 1

   average_predicted_scores = 0
   if num_ones > num_zeroes:
      average_predicted_score = 1
   else:
      average_predicted_score = 0

   features = {'average_scores': average,
               'prediction': average_predicted_score}
   return features

full = 0

def scrape1():
   subdirectories = chain(os.walk("Review1"),
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
   subdirectories = os.walk("Review2")

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

def append_reviews(review_set, binary_good_or_bad):
   words = []
   for review in review_set:
      cnt = 0
      for paragraph in review['review']:
         if len(paragraph) > 15: 
            if cnt == 0 and review['FOOD'] == binary_good_or_bad:
               for word in paragraph.split():
                  words.append(word)
            elif cnt == 1 and review['SERVICE'] == binary_good_or_bad:
               for word in paragraph.split():
                  words.append(word)
            elif cnt == 2 and review['VENUE'] == binary_good_or_bad:
               for word in paragraph.split():
                  words.append(word)
            if cnt == 3 and review['OVERALL'] == binary_good_or_bad:
               for word in paragraph.split():
                  words.append(word)
   return words

def naive_bayes_tuples_e1(review_set, good_words, bad_words):
   data = []
   for review in review_set:
      cnt = 0
      for paragraph in review['review']:
         if len(paragraph) > 15:
            if cnt == 0:
               review_tuple = (paragraph_features(paragraph, good_words, bad_words), review['FOOD'])
               data.append(review_tuple)
               cnt = cnt + 1
            elif cnt == 1:
               review_tuple = (paragraph_features(paragraph, good_words, bad_words), review['SERVICE'])
               data.append(review_tuple)
               cnt = cnt + 1
            elif cnt == 2:
               review_tuple = (paragraph_features(paragraph, good_words, bad_words), review['VENUE'])
               data.append(review_tuple)
               cnt = cnt + 1
            elif cnt == 3:
               review_tuple = (paragraph_features(paragraph, good_words, bad_words), review['OVERALL'])
               data.append(review_tuple)
               cnt = cnt + 1
   return data

def naive_bayes_tuples_e2(review_set, good_words, bad_words, classifier):
   data = []
   for review in review_set:
      review_tuple = (review_features(review, good_words, bad_words, classifier), review['OVERALL_RATING'])
      data.append(review_tuple)
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
  

tr = True


if __name__ == '__main__':
   '''
   #if tr:
   #   predict_author()
         
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
   print(remove_stopwords("Hello this is my sentence also us like the a and of but i on"))
   '''

   # List of reviews, which are dictionaries
   train = scrape1()
   test = scrape2()

   good_words = append_reviews(train, 1)
   good_words = [word.lower() for word in good_words if word.lower() not in sw.words("english")
                                                     and word.lower() != 'food' and word.lower() != 'service'
                                                     and word.lower() != 'venue' and word.lower() != 'restaurant'] 
   #good_words = [nltk.pos_tag([word])[0] for word in good_words]
   #good_words = [word for (word, pos) in good_words if pos == 'JJ']
   #good_words = common_words(good_words, 30)
   #print(common_words(good_words, 30))

   bad_words = append_reviews(train, 0)
   bad_words = [word.lower() for word in bad_words if word.lower() not in sw.words("english")
                                   and word.lower() != 'food' and word.lower() != 'service'
                                   and word.lower() != 'venue' and word.lower() != 'restaurant'] 
   #bad_words = [nltk.pos_tag([word])[0] for word in bad_words]
   #bad_words = [word for (word, pos) in bad_words if pos == 'JJ']
   #bad_words = common_words(bad_words, 30)
   #print(common_words(bad_words, 30))

   train_data = naive_bayes_tuples_e1(train, good_words, bad_words)
   test_data = naive_bayes_tuples_e1(test, good_words, bad_words)

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

   refsets = collections.defaultdict(set)
   testsets = collections.defaultdict(set)

   for i, (feats, label) in enumerate(test_data):
      refsets[label].add(i)
      observed = classifier.classify(feats)
      testsets[observed].add(i)

   print("Precision For Bad Rating: " + str(precision(refsets[0], testsets[0])))
   print("Recall For Bad Rating: " + str(recall(refsets[0], testsets[0])))
   print("F-measure For Bad Rating: " + str(f_measure(refsets[0], testsets[0])))
   print("Precision For Good Rating: " + str(precision(refsets[1], testsets[1])))
   print("Recall For Good Rating: " + str(recall(refsets[1], testsets[1])))
   print("F-measure For Good Rating: " + str(f_measure(refsets[1], testsets[1])))
   print(classifier.show_most_informative_features(20))