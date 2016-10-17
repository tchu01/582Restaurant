from bs4 import BeautifulSoup, NavigableString, Tag
import unicodedata
import re

regex = ['REVIEWER', 'NAME', 'ADDRESS', 'CITY', 'FOOD', 'SERVICE', 'VENUE', 'OVERALL']
regexs= [("b*'*REVIEWER:*(.*)'*", 'REVIEWER'),
		 ("b*'*NAME:*(.*)'*", 'NAME'),
		 ("b*'*ADDRESS:*(.*)'*", 'ADDRESS'),
		 ("b*'*CITY:*(.*)'*", 'CITY'),
		 ("b*'*FOOD:*(.*)'*", 'FOOD'),
		 ("b*'*SERVICE:*(.*)'*", 'SERVICE'),
		 ("b*'*VENUE:*(.*)'*", 'VENUE'),
		 ("b*'*OVERALL:*(.*)'*", 'OVERALL'),
		 ("b*'*RATING:*(.*)'*", 'OVERALL')]


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
	data_dict['review'] = p_stuff[ind - len(p_stuff):]
	p_stuff = [str(p) for p in p_stuff]
	#print(len(p_stuff))

	for i in range(0, len(p_stuff)):
		p_stuff[i] = p_stuff[i].replace('"', "'")
		p_stuff[i] = p_stuff[i].replace('\\n', '')

	#for p in p_stuff:
	#	print(p)

	for reg in regexs:
		#print('Reg[1]: %s' % reg[1])
		reg_found = False
		for line in p_stuff:
			match = re.match(reg[0], str(line))
			if match:
				reg_found = True
				data_dict[reg[1]] = match.group(1)
				break
		if not reg_found:
			reg_string = reg[1] + r'(:*)(.*?)REVIEWER|NAME|ADDRESS|CITY|FOOD|SERVICE|VENUE|OVERALL'
			match = re.match(reg_string, str(line))
			if match:
				data_dict[reg[1]] = match.group(1)
			else:
				data_dict[reg[1]] = None

	#print(data_dict)
	#if 'Nupur Garg' in data_dict['REVIEWER']:
	#	return None
	return data_dict

	#REVIEWER(:*)(.*?)REVIEWER|NAME|ADDRESS|CITY|FOOD|SERVICE|VENUE|OVERALL