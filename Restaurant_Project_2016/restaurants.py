import sys, os, random, nltk, re, collections, math
from os import listdir
from os.path import isfile, join
from itertools import chain
from nltk.corpus import stopwords as sw
from nltk.corpus import movie_reviews as mr
from nltk.metrics.scores import precision, recall, f_measure

import review_scraper as rs
import feature_extraction as fe

def paragraph_rating(train_data, test_data):
   print("Exercise 1")

   good_words = fe.append_reviews(train_data, 1)
   good_words = [word.lower() for word in good_words 
                 if word.lower() not in sw.words("english")
                 and word.lower() != 'food' and word.lower() != 'service'
                 and word.lower() != 'venue' and word.lower() != 'restaurant']
   bad_words = fe.append_reviews(train_data, 0)
   bad_words = [word.lower() for word in bad_words 
                if word.lower() not in sw.words("english")
                and word.lower() != 'food' and word.lower() != 'service'
                and word.lower() != 'venue' and word.lower() != 'restaurant']

   training = fe.naive_bayes_tuples_e1(train_data, good_words, bad_words)
   test = fe.naive_bayes_tuples_e1(test_data, good_words, bad_words)

   classifier = nltk.NaiveBayesClassifier.train(training)
   print("Accuracy: ",nltk.classify.accuracy(classifier,test))

   refsets = collections.defaultdict(set)
   testsets = collections.defaultdict(set)

   for i, (feats, label) in enumerate(test):
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
   return (classifier, good_words, bad_words)

def overall_rating(train_data, test_data, exercise1_classifier, good_words, bad_words):
   print("Exercise 3")

   training = fe.naive_bayes_tuples_e2(train_data, good_words, bad_words, exercise1_classifier)
   test = fe.naive_bayes_tuples_e2(test_data, good_words, bad_words, exercise1_classifier)

   classifier = nltk.NaiveBayesClassifier.train(training)
   print("Accuracy: ",nltk.classify.accuracy(classifier,test))

   refsets = collections.defaultdict(set)
   testsets = collections.defaultdict(set)

   for i, (feats, label) in enumerate(test):
      refsets[label].add(i)
      observed = classifier.classify(feats)
      testsets[observed].add(i)

   print("Precision For Average Score of 1: " + str(precision(refsets[1], testsets[1])))
   print("Recall For Average Score of 1: " + str(recall(refsets[1], testsets[1])))
   print("F-measure For Average Score of 1: " + str(f_measure(refsets[1], testsets[1])))
   print("Precision For Average Score of 2: " + str(precision(refsets[2], testsets[2])))
   print("Recall For Average Score of 2: " + str(recall(refsets[2], testsets[2])))
   print("F-measure For Average Score of 2: " + str(f_measure(refsets[2], testsets[2])))
   print("Precision For Average Score of 3: " + str(precision(refsets[3], testsets[3])))
   print("Recall For Average Score of 3: " + str(recall(refsets[3], testsets[3])))
   print("F-measure For Average Score of 3: " + str(f_measure(refsets[3], testsets[3])))
   print("Precision For Average Score of 4: " + str(precision(refsets[4], testsets[4])))
   print("Recall For Average Score of 4: " + str(recall(refsets[4], testsets[4])))
   print("F-measure For Average Score of 4: " + str(f_measure(refsets[4], testsets[4])))
   print("Precision For Average Score of 5: " + str(precision(refsets[5], testsets[5])))
   print("Recall For Average Score of 5: " + str(recall(refsets[5], testsets[5])))
   print("F-measure For Average Score of 5: " + str(f_measure(refsets[5], testsets[5])))
   '''
   print("Precision For Prediction of one: " + str(precision(refsets['one'], testsets['one'])))
   print("Recall For Prediction of one: " + str(recall(refsets['one'], testsets['one'])))
   print("F-measure For Prediction of one: " + str(f_measure(refsets['one'], testsets['one'])))
   print("Precision For Prediction of zero: " + str(precision(refsets['zero'], testsets['zero'])))
   print("Recall For Prediction of zero: " + str(recall(refsets['zero'], testsets['zero'])))
   print("F-measure For Prediction of zero: " + str(f_measure(refsets['zero'], testsets['zero'])))
   '''
   print(classifier.show_most_informative_features(20))

def predict_author(train_data, test_data):
   print("Exercise 4")
   training, test = fe.predict_authorship_classifier(train_data, test_data)
 
   classifier = nltk.NaiveBayesClassifier.train(training)
   print("Accuracy: ",nltk.classify.accuracy(classifier,test))

def phenomena(train_data, test_data, good_words, bad_words):
   print("Exercise 2")
   #ratio of adj:noun for good reviews and bad reviews
   good_words_pos = [nltk.pos_tag([word]) for word in good_words]
   bad_words_pos = [nltk.pos_tag([word]) for word in bad_words]
   good_nouns = 0
   good_adjs = 0
   bad_nouns = 0
   bad_adjs = 0
   for [(word,pos)] in good_words_pos:
      if pos == 'NN':
         good_nouns += 1
      if pos == 'JJ':
         good_adjs += 1   
   for [(word,pos)] in bad_words_pos:
      if pos == 'NN':
         bad_nouns += 1
      if pos == 'JJ':
         bad_adjs += 1

   print("Ratio of nouns to adjectives for good reviews: " + str(good_nouns) + '/' + str(good_adjs))
   print(good_nouns/good_adjs)
   print("Ratio of nouns to adjectives for bad reviews: " + str(bad_nouns) + '/' + str(bad_adjs))
   print(bad_nouns/bad_adjs)

   print()
   #top 50 words in good set not in bad set
   #top 50 words in bad set not in good set
   fdist_good = nltk.FreqDist(good_words).most_common(50)
   top_50_good = [word for (word,freq) in fdist_good]
   fdist_bad = nltk.FreqDist(bad_words).most_common(50)
   top_50_bad = [word for (word,freq) in fdist_bad]
   print("Top words in good set not in bad set (considering top 50): " + str(list(set(top_50_good) - set(top_50_bad))))
   print("Top words in bad set not in good set (considering top 50): " + str(list(set(top_50_bad) - set(top_50_good))))

   print()
   #look at reviews with rating of 1 and rating of 5 and find common words
   train_data_4s = [review for review in train_data if review['OVERALL_RATING'] == 4]
   train_data_2s = [review for review in train_data if review['OVERALL_RATING'] == 2]
   test_data_4s = [review for review in test_data if review['OVERALL_RATING'] == 4]
   test_data_2s = [review for review in test_data if review['OVERALL_RATING'] == 2]

   words_4 = fe.append_reviews_overall(train_data_4s)
   words_4 = [word.lower() for word in words_4 
              if word.lower() not in sw.words("english")
              and word.lower() != 'food' and word.lower() != 'service'
              and word.lower() != 'venue' and word.lower() != 'restaurant']

   words_2 = fe.append_reviews_overall(train_data_2s)
   words_2 = [word.lower() for word in words_2 
              if word.lower() not in sw.words("english")
              and word.lower() != 'food' and word.lower() != 'service'
              and word.lower() != 'venue' and word.lower() != 'restaurant']
   words_4t = fe.append_reviews_overall(test_data_4s)
   words_4t = [word.lower() for word in words_4t
               if word.lower() not in sw.words("english")
               and word.lower() != 'food' and word.lower() != 'service'
               and word.lower() != 'venue' and word.lower() != 'restaurant']
   words_2t = fe.append_reviews_overall(test_data_2s)
   words_2t = [word.lower() for word in words_2t
               if word.lower() not in sw.words("english")
               and word.lower() != 'food' and word.lower() != 'service'
               and word.lower() != 'venue' and word.lower() != 'restaurant']
   print("Common words in overall ratings of 4 (training): " + str(nltk.FreqDist(words_4).most_common(10)))
   print("Common words in overall ratings of 2 (training): " + str(nltk.FreqDist(words_2).most_common(10)))
   print("Common words in overall ratings of 4 (test): " + str(nltk.FreqDist(words_4t).most_common(10)))
   print("Common words in overall ratings of 2 (test): " + str(nltk.FreqDist(words_2t).most_common(10)))

def mean(nums):
   return float(sum(nums)) / max(len(nums), 1)

def rmse(predictions, targets):
   return math.sqrt((mean((predictions - targets) ** 2)))

full = 0

def main():
   test_data = []
   train_data = []
   if len(sys.argv) != 2:
      print("Usage: python3 restaurants.py <DATA_DIR>")
      return
   else:
      if sys.argv[1] == '-h':
         print("Usage: python3 restaurants.py <DATA_DIR>")
         print("Team Members: Timothy Chu and Sam Lakes")
         return
      else:
         path = sys.argv[1]


         # Find path to directory
         if os.path.isabs(sys.argv[1]) is False:
            path = os.path.abspath(sys.argv[1])

         # Check if directory has "training" and "test" directories
         if os.path.isdir(path + "/test") is True and os.path.isdir(path + "/training") is True:
            # Scrape training/test folders
            print("Found test and training directories")
            training_files = [path + "/training/" + f for f in listdir(path + "/training")
                                                      if isfile(join(path + "/training", f))
                                                      and f != '.DS_Store']
            testing_files = [path + "/test/" + f for f in listdir(path + "/test")
                                                 if isfile(join(path + "/test", f))
                                                 and f != '.DS_Store']

            for file_path in training_files:
               train_data.append(rs.scrape_page(file_path, None))

            for file_path in testing_files:
               test_data.append(rs.scrape_page(file_path, None))
         else:
            # Else ... create own test/training set
            print("No test and training directories... creating own test/training set")
            train_set = ["Review1","Review2", "Review3"]
            test_set = random.choice(train_set)
            train_set.pop(train_set.index(test_set))
            train_subs = [path + '/' + tset for tset in train_set]
            train_subs = chain.from_iterable([os.walk(i) for i in train_subs])
            for file_path in train_subs:
               if len(file_path[1]) == full:
                  matchName = re.match(r'(.*) (.*)', file_path[0])
                  reviewer_name = matchName.group(1).split('/')[-1] + ' ' + matchName.group(2).split('_')[0]
                  train_data.append(rs.scrape_page(file_path[0] + '/onlinetext.html', reviewer_name))
            train_data = [d for d in train_data if d]
            
            test_subs = os.walk(test_set)
            for file_path in test_subs:
               if len(file_path[1]) == full:
                  matchName = re.match(r'(.*) (.*)', file_path[0])
                  reviewer_name = matchName.group(1).split('/')[-1] + ' ' + matchName.group(2).split('_')[0]
                  test_data.append(rs.scrape_page(file_path[0] + '/onlinetext.html', reviewer_name))
            test_data = [d for d in test_data if d]

      (exercise1_classifier, good_words, bad_words) = paragraph_rating(train_data, test_data)
      phenomena(train_data, test_data, good_words, bad_words)
      overall_rating(train_data, test_data, exercise1_classifier, good_words, bad_words)
      predict_author(train_data, test_data)

if __name__ == '__main__':
   main()
