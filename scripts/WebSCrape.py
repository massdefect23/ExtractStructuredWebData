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

# NER: Places
"""
lets enrich the data.

1. get the place names
use NLP & NER to extract place names from the text.

NLP is a form of machine learning 
, algorithms use grammar and syntax rules to learn
relationships between words in text. Using that learning, NER
is able to understand the role that certain words play
within a sentence in a paragraph. 

Get coordinates from external API

"""

geo_api_key='12237b3149394a309fb9e7a4756b2a73'

def get_coords(content_list):
	iteration=1
	for i in range(len(content_list)):
		print('Getting coordinates',iteration,'/',len(content_list))
		trmp_list=[]
		text=content_list[i]

		# apply NER algorithm from ner-d

		doc=nlp(text)
		location=[x.text for x in doc.ents if x.label_ == 'GPE']
		location_dict=dict.fromkeys(location,0)
		location=list(location_dict)


		# querying the locations in open cage

		for l in location:
			try:
				request_url='https://api.opencagedata.com/geocode/v1/json?q={}&key={}'.format(l,geo_api_key)
				page=requests.get(request_url)
				data=page.json()
				for n in range(len(data)):

					if data['results'][n]['components']['country'] in mentioned_countries[i]:
						lat=data['results'][n]['geometry']['lat']
						lng=data['results'][n]['geometry']['lng']
						coordinates={'Location': l, 'Lat': lat, 'Lon': lng}
						temp_list.append(coordinates)
						break
					else:
						continue
			except Exception:
				continue
			coord_list.append(temp_list)
			iteration+=1


# NER: people names, rid of duplicates, validate 

def get_people(content_list):
	iteration=1

	# use NER to find names in text

	for i in range(len(content_list)):
		print('Getting people',iteration,'/',len(content_list))
		temp_list=[]
		text=content_list[i]
		doc=nlp(text)
		persons=[x.text for X in doc.ents if x.lable_ == "PERSON"]
		persons_dict=dict.fromkeys(persons,0)
		persons=list(persons_dict)


		full_names=[]
		for person in persons:
			for name in full_names:
				tokens=word_tokenize(name)
				for n in range(len(tokens)):
					if person==tokens[n]:
						final_names.append(name)

		for name in full_names:
			final_names.append(name)

		name_dict=dict.fromkeys(final_names,0)
		final_names=list(name_dict)
		valid_names=[]

		for name in final_names:
			page=requests.get('https://en.wikipedia.org/wiki/'+name)
			if page.status_code==200:
				valid_names.append(name)

		people.append(valid_names)
		iteration+=1

"""
keyword extraction: term frequency-inverse doc frequency
TF-IDF models measure how often a term or a word was used
in a single document, then compares that to its average use 
throughout the entire corpus of docments, then its likely that the term
represents a keyword unique to that specific document.

we create a 'bag of words', tracks every word used in 
every document. Then will count every usage - term frequency
then takes the common logarithm of every sentence in the 
document containing the term - inverse doc frequency.

These values are written as coordinates in a matrix,
which is then sorted to help us find the words most likely
to represent unique keywords for our document.
"""

def pre_process(text):
	text=text.lower()
	text=re.sub("</?.*?>"," <> ",text)
	text=re.sub("(\\d|\\W)+"," ",text)
	return text

# this maps matrices to coordinates 
# TF-IDF maps frequency scores to matrices,
# then need to be sorted to help find key words

def sort_coo(coo_matrix):
	tuples=zip(coo_matrix.col, coo_matrix.data)
	return sorted(tuples,key=lambda x:(x[1],x[0]), reverse=True)

# above is helper function that assists in sorting and selection
# of keywords once frequencies have been mapped to matrices
# this function specifically helps us choose most relevant 
# keywords based on TF-IDF statistics 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
	sorted_items=sorted_items[:topn]
	score_vals=[]
	feature_vals=[]

	for idx,score in sorted_items:
		fname=feature_names[idx]
		score_vals.append(round(score,3))
		feature_vals.append(feature_names[idx])

	results={}
	for idx in range(len(feature_vals)):
		results[feature_vals[idx]]=score_vals[idx]
	return results


# final function incorporates teh above helper functions,
# applies a tf-idf algorithm to the body of our text to 
# find keywords on frequency of usage.

def get_keywords(content_list):
	iteration=1
	processed_text=[pre_process(text) for text in content_list]
	stop_words=set(stopwords.words('english'))
	cv=CountVectorizer(max_df=0.85,stop_words=stop_words)
	word_count_vector=cv.fit_transform(processed_text)

	tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
	tfidf_transformer.fit(word_count_vector)

	feature_names=cv.get_feature_names()

	for i in range(len(processed_text)):
		print('Getting Keywords',iteration,'/',len(content_list))
		doc=processed_text[i]
		tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
		sorted_items=sort_coo(tf_idf_vector.tocoo())
		keys=extract_topn_from_vector(feature_names,sorted_items,10)
		keywords.append(list(keys.keys()))
		iteration+=1


# Topic modelling 
"""
One of the most common tasks in NLP is topic modelling. A form of 
clustering attempts to automatically sort documents into categories 
based on text content.

In this specific instance, we want to know what topics ISW covers.

"""

# vectorization
"""
k-means clustering algo to conduct topic modelling.
first use TF-IDF algo again to vectorize each doc.

vectorization is an ML term that refers to the transformation
of non numerical data into numerical spatial data
that the computer can use to conduct machine learning.
"""

# clustering
"""
once each cluster is complete, i save the number of each 
cluster (1-50) to a list of cluster_numbers and the 
keywords making up each cluster to a list of cluster_keywords
"""

















