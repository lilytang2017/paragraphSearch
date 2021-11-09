### Searching for legal documents at paragraph level

This repository contains the code and data for the **paper** "Searching for legal documents at paragraph level: Automating label generation and boosting of neural models of semantic similarity" by Li Tang and Simon Clematide of the University of Zurich, presented at the [Natural Legal Language Processing Workshop](https://nllpw.org) on the 10th of November, 2021 (Full text PDF at [ACL Anthology](https://aclanthology.org/2021.nllp-1.12/))  

**Abstract**: Searching for legal documents is a specialized Information Retrieval task that is relevant for expert users (lawyers and their assistants) and for non-expert users. By searching previous court decisions (cases), a user can better prepare the legal reasoning of a new case. Being able to search using a natural language text snippet instead of a more artificial query could help to prevent query formulation issues. Also, if semantic similarity could be modeled beyond exact lexical matches, more relevant results can be found even if the query terms don’t match exactly. For this domain, we formulated a task to compare different ways of modeling semantic similarity at paragraph level, using neural and non-neural systems. We compared systems that encode the query and the search collection paragraphs as vectors, enabling the use of cosine similarity for results ranking. After building a German dataset for cases and statutes from Switzerland, and extracting citations from cases to statutes, we developed an algorithm for estimating semantic similarity at paragraph level, using a link-based similarity method. When evaluating different systems in this way, we find that semantic similarity modeling by neural systems can be boosted with an extended attention mask that quenches noise in the inputs.

**[License](https://creativecommons.org/licenses/by/4.0/)**: Creative Commons Attribution 4.0 International License

**Code**: Python 3.7

**Run** the main script using 'python paragraphSearch.ref.py'. It will use the following files as input, during the run:

* casePar.IDs.v3.csv: case paragraphs with IDs and caseNames (unique for a case document)
* casePar.pairs.v3.csv: pairs of case paragraphs ('case pairs', see section 3.5 in the paper)
* tfidf.values.csv: tf-idf values for words in the case documents vocabulary, created by the script 'tfidf.calculation_cases.py'

Important parameters to change during a run are in lines 23-30, compare with Table 2 in the paper. 

**Additional files**:
* law.csv: statute (law) paragraphs, with law ID, article number and paragraph number (see section 3.1, statute paragraphs)
* triples.v3.csv: triples (see section 3.2 in the paper)







