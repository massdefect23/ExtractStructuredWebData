import requests
import nltk
import math
import re
import spacy
import regex as re
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json

# download nltk packages

from bs4 import BeautifulSoup
from nltk import *
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords

# most environments requires NER-D

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# initialize data fields for final dataset

dates=[]
titles=[]
locations=[]
people=[]
key_countries=[]
content_text=[]
links=[]
coord_list=[]
mentioned_countries=[]
keywords=[]
topic_categories=[]

# initializing cluster variables for topic modelling

cluster_keywords=[]
cluster_number=[]


# initialize NLP object using spacy
nlp = spacy.load("en_core_web_sm")


# extract hrefs

urls=['http://www.understandingwar.org/publications?page={}'.format(i) for i in range(179)]
hrefs=[]

def get_hrefs(page,class_name):
  page=requests.get(page)
  soup=BeautifulSoup(page.text,'html.parser')
  container=soup.find_all('div',{'class':class_name})
  container_a=container[0].find_all('a')
  links=[container_a[i].get('href') for i in range(len(container_a))]
  for link in links:
    if link[0]=='/':
      hrefs.append('http://www.understandingwar.org'+link)

for url in urls:
  get_hrefs(url,'view-content')


# extract data

def get_date(soup):
	try:
		data=soup.find('span',{'class':'submitted'})
		content=data.find('span')
		date=content.get('content')
		dates.append(date)
	except Exception:
		dates.append('')
		pass

# function to extract product title 
def get_title(soup):
	try:
		title=soup.find('h1',{'class':'title'}).contents
		titles.append(title[0])
	except Exception:
		titles.append('')
		pass

# function to extract text
def get_contents(soup):
	try:
		parents_blacklist=['[document]','html','head','style'\
		'script','body','div','a','section','tr','td','label'\
		'ul','header','aside',]
		content=''
		text=soup.find_all(text=True)

		for t in text:
			if t.parent.name not in parents_blacklist and len(t) > 10:
				content=content+t+' '

		content_text.append(content)
	except Exception:
		content_text.append('')
		pass


# NLP
# figure out what countries are referenced in product.

def get_countriers(content_list):
	iteration=1
	for i in range(len(content_list)):
		print('Getting countries',iteration,'/',len(content_list))
		temp_list=[]
		for word in word_tokenize(content_list[i]):
			for country in country_list:
				if word.lower().strip() == country.lower().strip():
					temp_list.append(country)

		counted_countries=dict(Counter(temp_list))
		temp_dict=dict.fromkeys(temp_list,0)
		temp_list=list(temp_dict)
		if len(temp_list)==0:
			temp_list.append('Worldwide')
		mentioned_countries.append(temp_list)

		# counting number mentions for each country
		# then checking if country is mentioned more than
		# the mean number of mentions.

		keywords=[]
		for key in counted_countries.keys():
			if counted_countries[key] > np.mean(list(counted_countries.values())):
				keywords.append(key)
		if len(keywords) != 0:
			key_countries.append(keywords)
		else:
			key_countries.append(temp_list)
		iteration+=1





















