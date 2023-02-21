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
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# most environments requires NER-D

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
