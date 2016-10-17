import random, operator, nltk
from nltk.corpus import stopwords as sw

# Removes stopwords from a paragraph
def remove_stopwords(paragraph):
   short = paragraph.split()
   short = [word for word in short if word not in sw.words("english")]
   return ' '.join(short) 

# Finds the average score of food, service, venue for each review
def average_total_score(review):
   return {'avg_score':((review['food_score'] + review['service_score'] + review['venue_score']) / 3),
           'overall_score':review['overall_score']}

# Creates a dictionary for all reviewers for one review number (ie: Review#1)
# and calculates their avg score and lists their overall score.
def review_scores_per_person(dictionary):
   scores_per_person = {}
   for key in dictionary:
      scores_per_person[key] = average_total_score(dictionary[key])
      
   return scores_per_person

# Creates a list of the n most common words in the text
# Can be used to extract common words from all review with score 3-5 or 0-2
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
   return sum([len(sent) for sent in sents]) / len(sents)      

def paragraph_features(paragraph):
   features = {'lexical_diversity':lexical_diversity(paragraph), 
               'average_sent_length':avg_sent_length(paragraph)} 
   return 0

if __name__ == '__main__':
   # TODO Testing helper functions
   text = ["hello", "my", "name", "is", "Tim", "Tim", "hello", "good", "bad", "horrible"]
   common = common_words(text, 2)
   print(common)
   
   print(lexical_diversity(text))
   
   print(count_good_keywords(text))
   print(count_bad_keywords(text))

   sents = [["hello", "tim"], ["bye", "tim", "chu"]]
   print(avg_sent_length(sents))

   print(remove_stopwords("Hello this is my sentence"))
