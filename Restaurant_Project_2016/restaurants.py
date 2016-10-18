import sys, os, random, nltk

def main():
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

         else:
            # Else ... create own test/training set
            print("No test and training directories... creating own test/training set")

if __name__ == '__main__':
   main()
