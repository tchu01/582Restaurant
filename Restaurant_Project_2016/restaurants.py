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
         print("Looking for folder: " + str(sys.argv[1]))
         path = sys.argv[1]

         # Find path to directory
         if os.path.isabs(sys.argv[1]) is True:
            print("True")
            print(sys.argv[1])
         else:
            print("False")
            print(os.path.abspath('.'))

         # Check if directory has "training" and "test" directories

         # Else ... create own test/training set

if __name__ == '__main__':
   main()
