import sys, os, random, nltk, re, collections
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

def predict_author(train_data, test_data):
   print("Exercise 4")

   training, test = fe.predict_authorship_classifier(train_data, test_data)

   classifier = nltk.NaiveBayesClassifier.train(training)
   print("Accuracy: ",nltk.classify.accuracy(classifier,test))


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
   good_words_movie = mr.words(categories=['pos'])
   bad_words_movie = mr.words(categories=['neg'])
   good_words_movie = [nltk.pos_tag([word]) for word in good_words_movie]
   print(good_words_movie)
   bad_words_movie = [nltk.pos_tag([word]) for word in bad_words_movie]
   print(bad_words_movie)


   print(test_data)

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
      overall_rating(train_data, test_data, exercise1_classifier, good_words, bad_words)
      predict_author(train_data, test_data)
      phenomena(train_data, test_data, good_words, bad_words)

if __name__ == '__main__':
   main()
