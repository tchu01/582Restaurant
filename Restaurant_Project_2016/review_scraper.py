from bs4 import BeautifulSoup
import unicodedata
import re

regex = ['REVIEWER', 'NAME', 'ADDRESS', 'CITY', 'FOOD', 'SERVICE', 'VENUE', 'OVERALL']

def scrape_page(page, reviewer):
	data_dict = {}
	soup = BeautifulSoup(open(page, 'r'), "html.parser")
	p_stuff = [p.get_text().encode('utf-8') for p in soup.find_all('p')]
	p_stuff = [element for element in p_stuff if len(element) > 3]
	#if len(p_stuff) == 1:
	#	p_stuff = [p.contents for p in soup.find_all('br')]
	#	print(p_stuff[0])

	data_dict['review'] = p_stuff[-4:]
	
	for p in p_stuff:
		print(p)
	return(p_stuff)

