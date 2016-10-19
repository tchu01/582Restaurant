import review_scraper as rs
from itertools import chain
import os
import re

#There is a random ass extra directory, this is sort of a fix
full = 0

def test_string():
	g = r'NAME: Arayas PlaceADDRESS: 10246 Main St cCITY: BellevueFOOD: 5SERVICE: 4VENUE: 5RATING: 5WRITTEN REVIEW:'
	z = 'NAME: Bob'
	regex = ['REVIEWER', 'NAME', 'ADDRESS', 'CITY', 'FOOD', 'SERVICE', 'VENUE', 'OVERALL']
	regex_pattern = "|".join(regex)
	reg = 'NAME:*(.*?)' + regex_pattern  
	#reg = "NAME:*(.*)"
	print(reg)
	mtch = re.match(reg, g)
	print(mtch.group(1))
	


def main():
	subdirectories = chain(os.walk("Review1"), 
						   os.walk("Review2"), 
						   os.walk("Review3"))

	#subdirectories = os.walk("Review2")

	data = []
	for path in subdirectories:
		if len(path[1]) == full:
			matchName = re.match(r'(.*) (.*)', path[0])
			data.append(rs.scrape_page(path[0] + '/onlinetext.html', 
				        matchName.group(1).split('/')[1] + 
				        ' ' + 
				        matchName.group(2).split('_')[0]))

	data = [d for d in data if d]

	for d in data:
		print(d)
	print(len(data))


if (__name__ == '__main__'):
	main()


#NAME:*(.*?)(REVIEWER|NAME|ADDRESS)
#NAME:*(.*)
