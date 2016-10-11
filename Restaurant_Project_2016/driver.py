import review_scraper as rs
from itertools import chain
import os
import re

#There is a random ass extra directory, this is sort of a fix
full = 0

def main():
	subdirectories = chain(os.walk("Review1"), 
						   os.walk("Review2"), 
						   os.walk("Review3"))

	subdirectories = os.walk("Review1")

	data = []
	for path in subdirectories:
		if len(path[1]) == full:
			matchName = re.match(r'(.*) (.*)', path[0])
			data.append(rs.scrape_page(path[0] + '\onlinetext.html', 
				        matchName.group(1).split('\\')[1] + 
				        ' ' + 
				        matchName.group(2).split('_')[0]))
	
if (__name__ == '__main__'):
	main()