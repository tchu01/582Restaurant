from bs4 import BeautifulSoup
import unicodedata
import re

regex = ['REVIEWER', 'NAME', 'ADDRESS', 'CITY', 'FOOD', 'SERVICE', 'VENUE', 'OVERALL']

class Review_Scraper:
	def scrape_page(self, page):
		soup = BeautifulSoup(open("%s" % page), "html.parser")
		print(soup.find_all('p'))