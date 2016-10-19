import sys, os, random, nltk
from itertools import chain
from os import listdir
from os.path import isfile, join

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
                                if isfile(join(path + "/training", f))]
            testing_files = [path + "/test/" + f for f in listdir(path + "/test")
                                if isfile(join(path + "/test", f))]

            for file_path in training_files:
              train_data.append(rs.scrape_page(file_path, None))

            for file_path in testing_files:
              test_data.append(rs.scrape_page(file_path, None))
         else:
            # Else ... create own test/training set
            print("No test and training directories... creating own test/training set")
            train_set = ["Review1","Review2", "Review3"]
            test_set = random.choice(rands)
            rands.pop(test_set)
            train_subs = list(chain.from_iterable(train_set))

            for file_path in train_subs:
              if len(file_path[1]) == full:
                matchName = re.match(r'(.*) (.*)', path[0])
                train_data.append(rs.scrape_page(file_path[0] + '/onlinetext.html',
                            matchName.group(1).split('/')[1] +
                            ' ' +
                            matchName.group(2).split('_')[0]))
            train_data = [d for d in train_data if d]

            for file_path in test_subs:
              if len(file_path[1]) == full:
                matchName = re.match(r'(.*) (.*)', path[0])
                test_data.append(rs.scrape_page(file_path[0] + '/onlinetext.html',
                            matchName.group(1).split('/')[1] +
                            ' ' +
                            matchName.group(2).split('_')[0]))
            test_data = [d for d in test_data if d]
      paragraph_rating(train_data, test_data)
      overall_rating(train_data, test_data)
      predict_author(train_data, test_data)
      phenomena(train_data, test_data)

if __name__ == '__main__':
   main()
