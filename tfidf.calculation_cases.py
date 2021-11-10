#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import re
import os
import spacy
from scipy import spatial
import csv
import sys
from unidecode import unidecode
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy.linalg as LA

print(os.getcwd())

inputdata_caseIDs = '/Users/lily/Documents/master.thesis/data/casePar.IDs.v3.csv'
inputdata_casePairs = '/Users/lily/Documents/master.thesis/data/casePar.pairs.v3.csv'
outputfile = '/Users/lily/Documents/master.thesis/data/tfidf.values.csv'

f = open(inputdata_caseIDs, 'r')
rows = f.read().split('\n')
f.close()

print("Loading caseIDs infile: rows:", len(rows))
caseTextdic = defaultdict(lambda: "")
for row in rows:
    cols = row.split('\t')
    #print(row)
    if len(cols) == 4:
        caseName = cols[1]
        caseText = cols[3]
        #print("caseName:", caseName, " => ", caseText[:50])
        caseTextdic[caseName] += caseText + "   " #merge all case paragraph texts from the same caseName for tfidf calculation

docs = []
for case in caseTextdic.keys(): # iterate through all the caseName and put the values (merged texts) into the documents list.
    docs.append(caseTextdic[case])
print("Documents loaded (merged paragraphs by caseName):", len(docs))


cv=CountVectorizer()
word_count_vector=cv.fit_transform(docs)
print("After using CountVectorizer, shape:", word_count_vector.shape, "(documents, vocab size)")

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)

# print idf values 
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"]) 
 
# sort ascending 
df_sorted = df_idf.sort_values(by=['idf_weights'],ascending=False)
print("Top scoring words:", df_sorted.head(20))
df_idf.to_csv(outputfile, index=True, sep='\t')
print("Ouput now in:", outputfile)

