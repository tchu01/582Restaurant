from bs4 import BeautifulSoup, NavigableString, Tag
import unicodedata
import re

regex = ['REVIEWER', 'NAME', 'ADDRESS', 'CITY', 'FOOD', 'SERVICE', 'VENUE', 'OVERALL']

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
	data_dict['review'] = p_stuff[-1 * ind:]
	print(len(p_stuff))
	for p in p_stuff:
		print(p)

	for reg in regex:
		reg_found = False
		for line in p_stuff:
			match = re.match("aasfasfsfasdf", str(line))
			if match:
				reg_found = True
				data_dict[reg] = match.group(1)
				break
		if not reg_found:
			data_dict[reg] = None
		
	return data_dict