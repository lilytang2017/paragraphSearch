### This Python 3 script is a modification of the paragraphSearch.py script on GitHub at 
### https://github.com/lilytang2017/paragraphSearch (code for the NLLP 2021 paper "Searching for Legal Documents at Paragraph Level: Automating Label Generation and Use of an Extended Attention Mask for Boosting Neural Models of Semantic Similarity")
### by Li Tang and Simon Clematide (Universität Zürich), PDF of the paper is at https://aclanthology.org/2021.nllp-1.12/
### The aim of the modification is to allow a search in statute paragraphs, using case paragraphs (from the same triples) as queries
### This statute paragraph search task and evaluation was not mentioned in the above paper, as it was fully implemented after the paper was published
### as described in the Master Thesis "Document Search at Paragraph Level" authored by Li Tang (December 2021, Universität Zürich), supervisor Dr. Simon Clematide

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
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging

inputdata = 'triples.sel2.csv' # input data: each line matches a case paragraph and a statute paragraph (1:1), totally 120 lines, with a diversity of legal topics and concepts, with case paragraph texts modified to hinder lexical matches (test a model's ability to model semantic similarity instead)
idf_values = '../NLLP_paper_GitHub/tfidf.values.csv' # tf-idf values for words in the case documents vocabulary, see tfidf.calculation_cases.v1.py

#### VSM model selection 
vec_algo = 'bert' # accepts: random, tfidf, bert, use, doc2vec (tf-idf vectorizer, BERT transformer, Universal Sentence Encoder, doc2vec from gensim)
bert_type = 'GermanBERT' # accepts: GermanBERT, DistilBERT (used, if vec_algo == bert); for bert_type = 'test' see lines 470ff. for more info
use_finetuned_GermanBERT = 1 # set to 0 to use standard GermanBERT (not finetuned, downloaded via the Internet). Set to 1 to use the fine-tuned model 'FT2-GermanBERT'
tfidf_vec_type = 'selfmade' # accepts: selfmade (self-coded, with several paragraphs as documents in idf value calculation), module (use the tfidfvectorizer module in sklearn, each paragraph counts as a document)

##### parameters to vary during an experiment, to compare different semantic matching models at paragraph level
idf_mask_threshold = 2 # e.g. 2; above this value words will be kept (attention mask), set word_minlen = 0 (if vec_algo == 'bert')
idf_vec_threshold = 2.42 # e.g. 2.42; above this value tokens will be used for vectorization for the tf-idf baseline vectorization (if vec_algo == 'tfidf')

##### keep stable during comparison, to stay close to the conditions described in the thesis text ######
topresnum = 3 # how many of the top hits are considered in the calculation of summary results, i.e. precision and recall
word_minlen = 0 # minimal lengh (in chars) so tokens / words will be kept (BERT attention mask)
paragraph_minlen = 0 # minimal length of a paragraph to be processed, e.g. 0
paragraph_maxtokens_w = 200 # maximal number of word tokens to be considered when modeling a paragraph, starting at index 0 position (if vec_algo == 'bert'), typical: 220

########################

if use_finetuned_GermanBERT:
	save_directory = "../finetuningGermanBERT/models/FT2_GermanBERT" # achieves 25% in this task
		# fine-tuning data: datasets/FT2_GermanBERT_data.csv - a dataset of 9832 examples, using method 'd' described in the thesis (including the redundancy-reducing approach)
	print("Will use the fine-tuned model 'FT1-GermanBERT")
else:
	save_directory = "" # use the standard GermanBERT model (not fine-tuned), achieves 20.83% when we tested it
	print("Will use the standard GermanBERT model (not fine-tuned)")



cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

f = open(inputdata, 'r')
triples = f.read().split('\n')
f.close()
print("Loaded", len(triples), "triples.") # all triples from the input file should now be loaded into 'triples'

idfval = defaultdict(lambda: 0.0)
idf_vec_val = defaultdict(lambda: 0.0)
f = open(idf_values, 'r')
lines = f.read().split('\n')
idfvalcount = 0
idfvalveccount = 0

for line in lines:
	cols = line.split('\t')
	token = cols[0].lower() # associate lowercased words with idf values stored in 'tfidf_values'
	if len(token) > 0:                                             
		idfvalue = round(float(cols[1]), 2)
		if vec_algo == 'tfidf':
			tokenlenmin = 7
		elif vec_algo == 'bert':
			tokenlenmin = 5
		else:
			tokenlenmin = 3
		if re.search(r'^[a-z]{3}', token) and len(token) >= tokenlenmin: # use a token filter (= 'Boosted NNB' VSM)
		#if len(token) > 0:                                              # do not use a token filter (= 'NNB' VSM)
			idfvalue = round(float(cols[1]), 2)
			idfval[token] = idfvalue
			idfvalcount += 1
			if idfvalue >= idf_vec_threshold: # for baseline vectorization using tf-idf algo ('seflmade' variant, see tfidf_vec_type)
				idf_vec_val[token] = idfvalue # store the actual tfidf value for this token
				idfvalveccount += 1

print("Total number of idf values for lowercased words loaded:", idfvalcount, " from input file lines:", len(lines))



if vec_algo == 'tfidf' or vec_algo == 'random':
	vector_vocab = []
	tfidf_vocab = len(idf_vec_val.keys())
	print("Vocabulary of tokens selected for vectorization:", tfidf_vocab, " based on tf-idf threshold:", idf_vec_threshold)
	for token in idf_vec_val.keys():
		vector_vocab.append(token) # create an ordered list for later vectorization
	vocab_len = len(vector_vocab)

	tvec = defaultdict(lambda: np.zeros((vocab_len,), dtype=np.float)) # default is the zero vector with the right size and dimension (numpy ndarray)
	i = 0
	for token in vector_vocab:
		val = idf_vec_val[token]
		np.put(tvec[token], i, val) # create a token vector to use later for paragraph-level vectorization
		i += 1
	print("  =>", i, "token vectors created.")



print("Extracting statutes (search collection) and case paragraphs (queries) from triples:")
triples_count = 0
searchCollection = []
queries = []
match = defaultdict(lambda: "")
for t in triples: 
    cols = t.split('\t')
    if len(cols) == 4: # checks if this line in triples contains all the data needed in 4 columns
        triples_count += 1
        #print("\n\n=============\nTRIPLE:", triples_count + 1)
        citation = cols[0]
        statuteRaw = cols[1]
        caseParText = cols[2]
        caseParText = re.sub(r' \(.+?\)',r'', caseParText) # get rid of parenthesis text to focus on main text (mostly citations)
        caseName = cols[3] # the unique identifier for the whole case document
        case_tokens = list(word_tokenize(caseParText))
        statute_tokens = list(word_tokenize(statuteRaw))
        # print("\nSTATUTE_p:", statuteRaw)
        # print("\nCASE_p:", caseParText)
        # print("\nTokens:", len(statute_tokens), len(case_tokens))
        searchCollection.append(statuteRaw)
        queries.append(caseParText)
        match[caseParText] = statuteRaw # this statute paragraph is then the expected match for this 'caseParText' in a search

print("\n\nProcessed triples:", triples_count, " matches:", len(match.keys()), " and the same number of queries and paragraphs to search.")




def preprocess_bert(slist):
	if bert_type == 'GermanBERT':
		from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification
		if save_directory:
			model = AutoModelForMaskedLM.from_pretrained(save_directory)	
			print("\n\nLoading checkpoint: ### ", save_directory, " ### \n\n")
		else:
			model = AutoModelForMaskedLM.from_pretrained("bert-base-german-cased")
		tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
	elif bert_type == 'DistilBERT':
		from transformers import DistilBertModel, DistilBertTokenizer 
		model = DistilBertModel.from_pretrained("distilbert-base-german-cased")
		tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-german-cased")
	else:
		print("Unrecognized BERT type!")
		quit()
	#wait = input("continue?")


	outlines = []
	filteredsent = []

	for p in slist:
		#print("\nOriginal:", p)
		tokens = p.split(" ")
		le = len(tokens)
		if le > paragraph_maxtokens_w:
			#print("\n##### too long for BERT? > 512 tokens incl. subwords #####")
			#print(p[:70])
			tokens = tokens[:paragraph_maxtokens_w]
		outline = ""
		for t in tokens:
			m = re.findall(r'[\w\-]+', t)
			try:
				w = m[0]
				#print("-t:", t, " w:", w, " len:", len(w))
				if len(w) > word_minlen:
					if idfval[w.lower()] == 0.0 or idfval[w.lower()] >= idf_mask_threshold:
						outline += w + " "
					else:
						outline += "_ "
				else:
					outline += "_ " # filtered tokens are replaced with "_" (2032 in vocab)
			except:
				pass
		outline.rstrip()
		#print("-proc:", outline)
		if len(p) > (paragraph_minlen - 1):
			outlines.append(outline)
			filteredsent.append(p)
	df = pd.DataFrame(outlines)
	#print("\ndf:\n", df[0])
	tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
	#print("\nTokenized:\n", tokenized)

	#wait = input("continue?")

	tup = (tokenized, filteredsent, tokenizer, model)
	return(tup)


def preprocess_others(slist): # apply only the paragraph_minlen filter here
	filteredsent = []
	for p in slist:
		if len(p) > (paragraph_minlen - 1):
			filteredsent.append(p)
	return(filteredsent)


def get_bert_vec(tokenized_data_all, max_len):
	ablation = 0 # for ablation experiments on the extended attention mask mechanism
	vecbatches = []
	num_paragraphs = tokenized_data_all.shape[0]
	print("\nVECTORIZATION (bert style): Received", num_paragraphs, "paragraphs. Creating batches of 10.")
	batches = []
	batch = []
	for b in range(0,num_paragraphs, 10): # get paragraph vectors, in batch mode (10 paragraphs at a time)
		print("-batch:", b, "-", b+9)
		batch = tokenized_data_all.iloc[b:b+10]
		padded = np.array([i + [0]*(max_len-len(i)) for i in batch.values])
		#print("\nnp array shape:", np.array(padded).shape)
		if ablation == 1:
			attention_mask = np.where((padded != 0), 1, 0) 
		else:
			attention_mask1 = np.where((padded != 0), 1, 0)    # replace the padded inputs with attention masks 0
			attention_mask2 = np.where((padded != 2032), 1, 0) # replace filtered tokens ("_", 2032) with attention mask 0 as well
			attention_mask = attention_mask1 * attention_mask2 # combine both (keep 1 only where both values were 1)
		#print("Attention mask shape:", attention_mask.shape)
		#print("Mask\n", attention_mask[:5])
		input_ids = torch.tensor(padded)  
		attention_mask = torch.tensor(attention_mask)
		with torch.no_grad():
			last_hidden_states = model(input_ids, attention_mask=attention_mask)
		vec = last_hidden_states[0][:,0,:].numpy()

		#print("vec shape:", vec.shape)
		#print(vec[:20])
		vecbatches.append(vec)
	print("\n### Paragraph embedding batches created, type:", type(vec))
	return vecbatches

def get_tfidf_vec(selsents):
	print("\nVECTORIZATION (tf-idf style): Paragraphs in input:", len(selsents), " Creating batches of 10.")

	if tfidf_vec_type == 'selfmade':
		print("  generating paragraph vectors using stored multi-paragraph idf values in:", idf_values)
	
		#print("  with tf-idf threshold:", tfidf_vec_threshold)
		batchindex = 0
		vecbatch = []
		vecbatches = []
		for p in selsents: # go through the list of paragraphs to vectorize
			pvec = np.zeros((vocab_len,), dtype=np.float) # initialize the paragraph vector with the right size and zeros
			#print("-paragraph:", p[:50])
			tokens = list(word_tokenize(p))
			#found = 0
			#foundtokens = []
			for t in tokens:
				t = t.lower()
				if idf_vec_val[t] >= idf_vec_threshold:
					#print(" -t:", t, " ### ", tfidf_vec_val[t])
					pvec += tvec[t] # add token vector with tf-idf value 
					#found += 1
					#positives = len(np.where(pvec > 0)[0].tolist())
					#print("  sum:", np.sum(pvec), " total:", found, " positives:", positives)

				else:
					#print(" -t:", t)
					pass
			vecbatch.append(pvec)
			#print("\nFound:", found, "     Found unique tokens:", len(set(foundtokens)))
			batchindex += 1
			if batchindex > 9: # reached the end of the batch
				batchindex = 0
				vecbatches.append(vecbatch) # add this batch of 10 vectors to the list of batches
				#print("Created # vectors:", len(vecbatch))
				vecbatch = []
		return vecbatches
				#wait = input("Continue?")

	elif tfidf_vec_type == 'module':
		print("  generating paragraph vectors using paragraph-level tf-idf values using sklearn's tfidfvectorizer:")
		vocab_t = vector_vocab # vocabulary parameter used (else: do not use 'vocabulary' parameter in the line below!)
		#tfidfvectorizer = TfidfVectorizer(input='content', encoding='utf-8', strip_accents='unicode', lowercase=True, analyzer='word', vocabulary=vocab_t, binary=False, smooth_idf=True, sublinear_tf=False, max_features=300)
		tfidfvectorizer = TfidfVectorizer(input='content', lowercase=True, vocabulary=vocab_t, analyzer='word', binary=True)
		tfidf_wm = tfidfvectorizer.fit_transform(selsents)
		tfidf_tokens = tfidfvectorizer.get_feature_names()
		print("tokens:", len(tfidf_tokens))
		df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
		
		print("module output:\n", df_tfidfvect)

		vecbatches = []
		for batchindex in range(0,len(selsents),10): # go through the list of paragraphs to vectorize
			batchstart = batchindex
			batchend = batchindex + 9
			if batchend > (len(selsents) - 1):
				batchend = len(selsents) - 1 

			print("\n-batch:", batchstart, " - ", batchend)
			df = df_tfidfvect[batchstart:batchend]
			#print(df)
			vecbatch = df.to_numpy()
			vecbatches.append(vecbatch) # add this batch of 10 vectors to the list of batches
		
		#wait = input("Continue?")
		return vecbatches


		

def get_use_vec(selsents):
	print("\nVECTORIZATION (USE style): Paragraphs in input:", len(selsents), " Creating batches of 10.")
	import spacy
	usevec = spacy.load('xx_use_md')
	batchindex = 0
	vecbatch = []
	vecbatches = []
	i = 0
	for p in selsents: # go through the list of paragraphs to vectorize 
		p = p[:800]
		pvec = usevec(p).vector
		i += 1
		if i % 10 == 0:
			print("  ###### vectors created:", i)
		vecbatch.append(pvec)
		#print("\nFound:", found, "     Found unique tokens:", len(set(foundtokens)))
		batchindex += 1
		if batchindex > 9: # reached the end of the batch
			batchindex = 0
			vecbatches.append(vecbatch) # add this batch of 10 vectors to the list of batches
			#usb_sim = round(np.inner(vecbatch[0], vecbatch[1]), 6)
			#print("- dot product test with first 2 vectors in this batch of 10:", usb_sim)
			vecbatch = []
			#wait = input("Continue?")
	return vecbatches

def get_doc2vec_vec(selsents):

	#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

	tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(selsents)]
	#print("Tagged Data:", tagged_data[:50])
	max_epochs = 12
	doc2vec_model = Doc2Vec(vector_size=200, window=5, min_count=1, workers=1, epochs=max_epochs, dm=0, seed=1, max_vocab_size=800, sample=2, negative=6, hs=0, ns_exponent=0.75, dm_mean=1, dm_concat=1)
	doc2vec_model.build_vocab(tagged_data)
	#wait = input("continue?")
	print("Training doc2vec model.")
	for epoch in range(max_epochs):
		print('  iteration {0}'.format(epoch))
		doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=max_epochs)
		doc2vec_model.alpha -= 0.0005
		doc2vec_model.min_alpha = doc2vec_model.alpha


	print("\nVECTORIZATION (doc2vec style): Paragraphs in input:", len(selsents), " Creating batches of 10.")
	batchindex = 0
	vecbatch = []
	vecbatches = []
	i = 0
	for p in selsents: # go through the list of paragraphs to vectorize 
		p = p[:900]
		words = list(word_tokenize(p.lower()))
		#print("words:", words)
		pvec = doc2vec_model.infer_vector(words)
		#print("pvec type:", type(pvec), "dim:", pvec.shape)
		#wait = input("continue?")
		i += 1
		if i % 10 == 0:
			print("  ###### vectors created:", i)
		vecbatch.append(pvec)
		#print("\nFound:", found, "     Found unique tokens:", len(set(foundtokens)))
		batchindex += 1
		if batchindex > 9: # reached the end of the batch
			batchindex = 0
			vecbatches.append(vecbatch) # add this batch of 10 vectors to the list of batches
			#usb_sim = round(np.inner(vecbatch[0], vecbatch[1]), 6)
			#print("- dot product test with first 2 vectors in this batch of 10:", usb_sim)
			vecbatch = []
			#wait = input("Continue?")
	return vecbatches


def get_sbert_vec(selsents):
	print("\nVECTORIZATION (sentence-BERT style): Paragraphs in input:", len(selsents), " Creating batches of 10.")
	batchindex = 0
	vecbatch = []
	vecbatches = []
	i = 0
	for p in selsents: # go through the list of paragraphs to vectorize 
		p = p[:800]
		pvec = model.encode([p])[0]
		i += 1
		if i % 10 == 0:
			print("  ###### vectors created:", i, " type:", type(pvec), pvec.shape)
		vecbatch.append(pvec)
		#print("\nFound:", found, "     Found unique tokens:", len(set(foundtokens)))
		batchindex += 1
		if batchindex > 9: # reached the end of the batch
			batchindex = 0
			vecbatches.append(vecbatch) # add this batch of 10 vectors to the list of batches
			#usb_sim = round(np.inner(vecbatch[0], vecbatch[1]), 6)
			#print("- dot product test with first 2 vectors in this batch of 10:", usb_sim)
			vecbatch = []
			#wait = input("Continue?")
	return vecbatches
		

if vec_algo == 'bert' and (bert_type == 'DistilBERT' or bert_type == 'GermanBERT' or bert_type == 'SPECTER'):

	print("\nPreprocessing for BERT:")
	#collection_tokenized = preprocess(collection)
	collection_tokenized, filteredsent, tokenizer, model = preprocess_bert(searchCollection)
	#print("tokenized:", collection_tokenized[:9])
	tokenized_decoded = collection_tokenized.apply((lambda x: tokenizer.decode(x)))
	#print("Decoded:\n", tokenized_decoded)
	#print("Type:", type(tokenized_decoded))
	t = tokenized_decoded.to_list()
	segword_count = defaultdict(lambda: 0)
	word = ""
	for p in collection_tokenized:
		p_words = []
		for t in p:
			t_decoded = tokenizer.decode(t)
			if re.search('^##', t_decoded):
				#print("-token:", t, "=>", t_decoded)
				word += t_decoded
			else:
				#print("-token:", t, "=>", t_decoded, "   w:", word)
				if len(word) > 12:
					p_words.append(word)
					segword_count[word] += 1 # count the frequency of this subword-segmented word, e.g. Sozial##versicherungs##gericht
				word = t_decoded
		#print("\n-subword seg:", " ".join(p_words))
		#print("\n-p:", tokenizer.decode(p))
	#wait = input("continue?")



	#print("as list:", t)
	max_len = 0 # padding
	for i in collection_tokenized.values:
		if len(i) > max_len:
			max_len = len(i) # determine max_len
	#print("\nMax_len:", max_len)
	print("Max len:", max_len)

	print("\nVectorize data paragraphs using an attention mask on padded data.")
	sent_vec_batches = get_bert_vec(collection_tokenized, max_len)

elif vec_algo == 'tfidf':
	filteredsent = preprocess_others(searchCollection)
	sent_vec_batches = get_tfidf_vec(filteredsent)
	#print("  created # of batches with 10 vectors each:", len(sent_vec_batches))

elif vec_algo == 'use':
	filteredsent = preprocess_others(searchCollection)
	sent_vec_batches = get_use_vec(filteredsent) 	

elif vec_algo == 'doc2vec':
	filteredsent = preprocess_others(searchCollection)
	sent_vec_batches = get_doc2vec_vec(filteredsent)

elif vec_algo == 'random':
	filteredsent = preprocess_others(searchCollection) 
	sent_vec_batches = get_tfidf_vec(filteredsent) # those vectors will not be used to calculate any vector similarity, in 'random' baseline mode

else:
	print("Check vec_algo flag, could not find a valid value. Value was:", vec_algo)
	sys.exit()





print("Now vectorizing QUERIES ######\n")
origqueries = queries

if vec_algo == 'bert' and (bert_type == 'DistilBERT' or bert_type == 'GermanBERT' or bert_type == 'SPECTER'):
	queries_tokenized, queries, bert_tok, bert_model = preprocess_bert(queries) # tokenized, filteredsent, tokenizer, model) #######

	#i = 0
	#for q in queries_tokenized:
		#print("- query tokens:", len(q), "   ", origqueries[i][:90])
		#i += 1
	#print("-shape:", queries_tokenized.shape)

	print("\nMax len in dataset was:", max_len)
	qbatches = get_bert_vec(queries_tokenized, max_len)

else: # non-BERT mode, e.g. tf-idf vectorization or USE, or sentence-BERT like transformers 
		if vec_algo == 'tfidf' or vec_algo == 'random':
			qbatches = get_tfidf_vec(queries)
		elif vec_algo == 'use':
			qbatches = get_use_vec(queries)
		elif vec_algo == 'doc2vec':
			qbatches = get_doc2vec_vec(queries)


#print("Query vectors shape, batch 0:", qbatches[0].shape)

print("\nSEARCH: Comparing query paragraph vectors with search collection paragraph vectors:\n")
print("  Loaded paragraphs to search:", len(filteredsent))

results = []
i = 0
for batch in sent_vec_batches:
	j = 0
	for query_vec in qbatches: # handle multiple batches of queries (10 each)
		for q in query_vec:
			for batchrow in range(0,10):
				try:
					if vec_algo == 'bert' or vec_algo == 'tfidf' or vec_algo == 'use' or vec_algo == 'doc2vec':
						cosine = round(float(cos(torch.tensor(batch[batchrow]),torch.tensor(q))), 4)
					elif vec_algo == 'random':
						cosine = round(random.randint(0,1000) / 1000, 3) # generate a random number between 0 and 1 with 3 positions after the comma, e.g. 0.342
						#print("- random:", randomnum)
					#else:
						#cosine = round((np.inner(batch[batchrow],q)/2000), 4) # USE: use np.inner to calculate the inner product of both vectors (not normalized)
					sentencerow = i + batchrow
					#print("\nBatch starts", i, "row:", batchrow, ":", filteredsent[sentencerow])
					#print("cosine:", sentencerow, ":", cosine)
					#print("-Q:", j, ":", queries[j])
					tup = (cosine, j, filteredsent[sentencerow])
					results.append(tup)
				except:
					cosine = 0
			j += 1 # query index, starting with 0
	i += 10
	if i % 50 == 0:
		print("  ### Datasets searched:", i)
		#wait = input("continue?")

print("# of cosine values calculated:", len(results))

print("\n\n================== Top results (only pairs shown): =================\n")
df = pd.DataFrame(results, columns=['cos','q','result'])
df.sort_values('cos', ascending=False, inplace=True)
print("  sorted, for determining top results with highest cosine.")

j = 0
resultssummary = ""
totalhits = 0
recalls, precisions = [], []
queryNum = 0

for q in queries:
	r = df[df['q'] == j]
	print("\n\n========= Results for casePar QUERY: ========== \n\n", queries[j], " \n")
	#lawIDs = defaultdict(lambda: "")
	correctMatch = match[q]
	print("- CORRECT Match:", correctMatch)
	#lawIDs[cp[0]] = cp[1]
	#print("=== Cited law was:", citedlaw[j], " \n")
	resultlist = r[['result']].head(topresnum).values.tolist()
	coslist = r[['cos']].head(topresnum).values.tolist()
	matchhits, otherhits = 0, 0
	matchpos = 1
	matchhitpos = 0
	for res in resultlist:
		if res[0] == correctMatch and matchhits == 0:
			print("\n####MATCH###: ", res[0][:150])
			matchhits += 1
			matchhitpos = matchpos
		else:
			print("\n-> ", res[0][:150])
			otherhits += 1
		matchpos += 1
	print("\ncos:", coslist)
	allhits = matchhits + otherhits
	totalhits += matchhits
		#recallPercent = round((matxhhits / maxmatches) * 100, 3)
	#precisionPercent = round(((matchhits / allhits) * 100), 3) # precision in %  for this query, among top n hits
	#recalls.append(recallPercent)
	#precisions.append(precisionPercent)

	resultssummary += str(matchhits) + " at pos. " + str(matchhitpos) + "   " + q[:30] + "\n"
	j += 1


print("\n\n===================RESULTS SUMMARY=============================\n\n", resultssummary, "\n=====================================")
avghits = round(totalhits / len(searchCollection), 3)
print("\n==> Totally found", str(totalhits), "hits in", str(len(searchCollection)), " queries.")

#avgrecall = round(sum(recalls) / len(recalls), 3)
avgmatches = round((totalhits * 100) / len(searchCollection), 3)
print(" Average matches in top", topresnum, ":", avgmatches, "%") 









