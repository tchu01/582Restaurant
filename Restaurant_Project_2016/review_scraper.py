# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup, NavigableString, Tag
import unicodedata
import re

#("b*'*(NAME|Name)+:*(.*)'*", 'NAME')


escapes = ''.join([chr(char) for char in range(1, 32)])
regex = ['REVIEWER', 'NAME', 'ADDRESS', 'CITY', 'FOOD', 'SERVICE', 'VENUE', 'OVERALL']
regex_format = ['Reviewer', 'Name', 'Address', 'City', 'Food', 'Service', 'Venue', 'Overall']
regexs= [("b*'*NAME:*(.*)'*", 'NAME'),
		 ("b*'*ADDRESS:*(.*)'*", 'ADDRESS'),
		 ("b*'*CITY:*(.*)'*", 'CITY'),
		 ("b*'*FOOD:*(.*)'*", 'FOOD'),
		 ("b*'*SERVICE:*(.*)'*", 'SERVICE'),
		 ("b*'*VENUE:*(.*)'*", 'VENUE'),
		 ("b*'*OVERALL:*(.*)'*", 'OVERALL'),
		 ("b*'*RATING:*(.*)'*", 'RATING')]


def scrape_page(page, reviewer):
	data_dict = {}
	soup = BeautifulSoup(open(page, 'r', encoding='utf8'), "html.parser")
	p_stuff = [p.get_text().encode('utf-8') for p in soup.find_all('p')]
	p_stuff = [element for element in p_stuff if len(element) > 3]

	if len(p_stuff) == 1:
		#go through entire review again, separate everything by br tags
		p_stuff = []
		p_stuff.append(soup.find("p").contents[0].strip())
		for br in soup.find_all('br'):
			nxt = br.next_sibling
			if not (nxt and isinstance(nxt, NavigableString)):
				continue
			next2 = nxt.next_sibling
			if next2 and isinstance(next2, Tag) and next2.name == 'br':
				text = str(next).strip()
				if text:
					p_stuff.append(nxt.encode('utf-8'))
	elif len(p_stuff) == 0:
		#go through entire review again, separate everything by \n
		p_stuff = [line.encode('utf-8') for line in str(soup).split('\n') if len(line) > 3]
		p_stuff[-1] = str(p_stuff[-1]).replace("</body></html>", "")
	elif len(p_stuff) > 13:
		return None

	#change to "find WRITTEN REVIEW using regex and then include every line break on and after it"
	#put those inside of the dictionary with key "review", which is a list of 4 paragraphs
	ind = next(i for i, string in enumerate(p_stuff) if 'WRITTEN REVIEW' in str(string))
	p_stuff[ind] = str(p_stuff[ind]).replace('WRITTEN REVIEW', "")
	regex_pattern = "|".join(regex)
	p_stuff = [str(p) for p in p_stuff]

	for i in range(0, len(p_stuff)):
		p_stuff[i] = p_stuff[i].replace('"', "'")
		p_stuff[i] = p_stuff[i].replace('\\n', '')

	#for p in p_stuff:
	#	print(p)

	data_dict['REVIEWER'] = reviewer
	for reg in regexs:
		#print('Reg[1]: %s' % reg[1])
		reg_found = False
		for line in p_stuff:
			#print(line)
			match = re.match(reg[0], str(line))
			if match:
				reg_found = True
				data_dict[reg[1]] = match.group(1)
				break
		if not reg_found:
			data_dict[reg[1]] = None

	if 'Nupur' in data_dict['REVIEWER']:
		return None
	return clean_dict(data_dict, p_stuff[ind - len(p_stuff):])


def clean_dict(d_dict, dict_rev):
	if d_dict['RATING'] != None:
		d_dict['OVERALL'] = d_dict['RATING']
	d_dict.pop('RATING')

	d_dict = {k:v.strip().replace("'", "").replace("\\xc2\\xa0", "").replace("\\xe2\\x80\\x99", "`") for k, v
	          in d_dict.items() if v}
	
	d_dict['OVERALL'] = float(d_dict['OVERALL'])
	d_dict['FOOD'] = float(d_dict['FOOD'])
	d_dict['SERVICE'] = float(d_dict['SERVICE'])
	d_dict['VENUE'] = float(d_dict['VENUE'])

	d_dict['OVERALL'] = binary(d_dict['OVERALL'])
	d_dict['FOOD'] = binary(d_dict['FOOD'])
	d_dict['SERVICE'] = binary(d_dict['SERVICE'])
	d_dict['VENUE'] = binary(d_dict['VENUE'])

	#\\xc3\\xb1


	d_dict['review'] = [x.replace("\\xe2\\x80\\x99", "`")for x in remove_b(dict_rev)]
	d_dict['review'] = [x.replace("\\xc2\\xa0", "") for x in d_dict['review']]
	d_dict['review'] = [x.replace("\\xc3\\xb1", "n") for x in d_dict['review']]
	d_dict['review'] = [x.replace("\\xe2\\x80\\xa6", "...") for x in d_dict['review']]
	d_dict['review'] = [x.replace("\\xe2\\x80\\x93", "-") for x in d_dict['review']]

	d_dict['review'] = remove_bad_elem(d_dict['review'])
	#for par in d_dict['review']:
	#	print("Len: %s  Par: %s\n" % (len(par), par))

	return d_dict

def binary(item):
	if item < 4:
		return 0
	else:
		return 1

def remove_b(lst):
	return [x.replace("b'", "'") for x in lst]


#I will do this tomorrow in order to get rid of a bad paragraph that is in most reviews
def remove_bad_elem(lst):
	return [x for x in lst if len(x) > 10]

	#REVIEWER(:*)(.*?)REVIEWER|NAME|ADDRESS|CITY|FOOD|SERVICE|VENUE|OVERALL


#wants all training done so he can run everything separate from each other
