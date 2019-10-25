# Ian Russell (Group 8)
# Advanced Data Mining
# October 25, 2019
# Assignment 3
# Question 5


# !!!!!! UPDATE PATH TO DATA FOLDER HERE !!!!!!!!!!
# Directory path to data folder
path = '/home/ian/Dropbox/School/Current_courses/Data_Mining/assignment_3/Data'

# !!!!!! SET 'fresh_data' TO TRUE IF DATA SET IS NEW (no matrix.csv or wordCounts.txt files) !!!!!!!!!
fresh_data = False


# import libraries
import re
import os
import pandas as pd
import ast
import csv
import numpy as np
import queue
import time

from progressbar import ProgressBar
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from num2words import num2words


"""
Document and TFIDFVector classes 
build a document 
and vector object for
some document in data set. 
These classes provide
encapsulation for processed data 
and for readability.
"""
class Document:
    
    def __init__(self, filename = '', path = '', text = ''):

        self.filename = filename
        self.path = path
        self.text = text
        
    def process_query(self):
        
        
        stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        porter = PorterStemmer()
        
        query = self.text
        
        # Tokenize
        query = re.sub(r'\W+', ' ', query).split()
        # apply lower case 
        query = [x.lower() for x in query]
        # remove stop words
        query = [i for i in query if not i in stop_words]
        # remove single characters
        query = [i for i in query if len(i) > 1]
        # stemming
        tmp = []
        for word in query:
            tmp.append(porter.stem(word))
        query = tmp

        # convert numbers
        index = 0
        for word in query:
            if word.isdigit():
                query[index] = num2words(word)
            index += 1

        return query

        
    def pre_process(self):
        """
        Pre-processor function to remove and tokenize indivual document
        object for prepartion of analysis. Returns processed data.
        """
        
        stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        porter = PorterStemmer()

        
        with open(os.path.join(path, self.filename), encoding="utf8", errors="surrogateescape") as text:

            data = text.read()
            
            # Tokenize
            data = re.sub(r'\W+', ' ', data).split()
            # apply lower case 
            data = [x.lower() for x in data]
            # remove stop words
            data = [i for i in data if not i in stop_words]
            # remove single characters
            data = [i for i in data if len(i) > 1]
            # stemming
            tmp = []
            for word in data:
                tmp.append(porter.stem(word))
            data = tmp
            
            # convert numbers
            index = 0
            for word in data:
                if word.isdigit():
                    data[index] = num2words(word)
                index += 1
            
            print("Document ready! " + str(self.filename))
            
            return data

class TFIDFVector:
    def __init__(self, document, N, DF):
        """
        Creates document vector to compute TF-IDF.
        For document vectorization, document should
        be a dictionary with word counts. For a query
        document should be in a string format.
        """
        self.document = document
        self.N = N
        self.DF = DF
        
        
    def vectorize(self):
        """
        Function generates TF-IDF Vector for
        individual document in corpus.
        """
        tf_idf_vector = {}

        for term in self.document:
            # term frequency
            count = self.document[term]
            tf = count/len(self.document)
            # document frequency
            df = sum(self.DF[term])
            # inverse document frequency
            idf = np.log(self.N/(df + 1))
            # TF-IDF
            tf_idf = tf*idf
            save = {term : tf_idf}
            tf_idf_vector.update(save)
            
        return tf_idf_vector
    
    def query_vectorize(self):
        """
        Function mirrors vectorize function
        but is modified for string inputs to 
        enable query functionality.
        """
        
        tf_idf_vector_query = {}
       
        for term in self.document:
            count = counter(self.document, term)[1]
            tf = count/len(self.document)
            # document frequency
            df = sum(self.DF[term])
            # inverse document frequency
            idf = np.log(self.N/(df + 1))
            # TF-IDF
            tf_idf = tf*idf
            save = {term : tf_idf}
            tf_idf_vector_query.update(save)
            
        return tf_idf_vector_query
    

def main(directory):
    """
    Main driver function, takes in directory and returns term counts for
    each individual document.
    """
    docs = []
    for entry in entries:
        docs.append(Document(entry, path))

    processed = []

    print('Processing documents...')
    print()
    for document in docs:
         processed.append(document.pre_process())
    
    processed_counts = termCounts(processed)
    
    with open('wordCounts.txt', 'w') as file:
        file.write(json.dumps(processed_counts))
    
    return processed_counts


def counter(lst, term): 
    """
    Utility function to count redundant occurences
    in a list.
    """
    count = 0
    for ele in lst: 
        if (ele == term): 
            count = count + 1
    return term, count


def termCounts(corpus):
    """
    Takes list of pre-processed documents (corpus) and returns individual
    term counts for each document as list of dictionaries.
    """
    count = 0
    termCount_corpus = []
    print()
    print("Counting... May take several minutes!")
    print()
    pbar = ProgressBar()
    print()
    for document in pbar(corpus):
        termCounts_doc = {}
        tmp = []
        for term in document:
            tmp.append(counter(document,term))
        terms = set(tmp)
        terms = list(terms)
        termCounts_doc.update(terms)
        termCount_corpus.append(termCounts_doc)
        count += 1
    return termCount_corpus

    


def universe_size(data):
    """
    Utility function to compute and return
    term universe size 'N'.
    """
    N = 0
    for doc in data: 
        n=0
        for term in doc:
            count = doc[term]
            n += count
        N += n
    return N
        
def document_frequency(data):
    """
    Utility function to compute document frequency for 
    all terms in pre-processed corpus.
    """
    DF = {}
    for i in range(len(data)):
        tokens = data[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
    return DF



def cosineSim(data):
    """
    Function to compute the pairwise cosine similarity
    throughout all documents. Produces an n x n matrix
    saved as a csv. n is number of documents. 
    
    """
    
    start_time = time.time()
    
    # Initialize all tf-idf vectors
    start_vectors  = []

    print('Initializing...')
    print()
    
    # Append each tfd-idf vector to start_vectors (all docs)

    pbarCos1 = ProgressBar()
    for i in pbarCos1(range(len(data))):
        start_vectors.append(TFIDFVector(data[i], universe_size(data), document_frequency(data)).vectorize())

    # Queue up vectors for comparison
    to_compare = queue.Queue(maxsize = len(data))
    cols = ["col" + str(i) for i in range(len(data))]

    for vector in start_vectors:
        to_compare.put(vector)

    print("Vectors ready")
    print()
    print("Working...")
    tick = 0
    
    # Start comparisons
    sims = pd.DataFrame()
    
    # Outer loop handles queued vectors

    print('WARNING: This can take up 90 minutes')

    pbarLong = ProgressBar()
    for vector in pbarLong(start_vectors):
        base = to_compare.get()
        print()
        column = []
        
        # Inner loop compares each vector in data to current vector in the queue
        for i in range(len(data)):
            
            # Convert pair of vectors to data frame
            comparison = pd.DataFrame([base, start_vectors[i]])
            
            # Replace non-present words from opposing document with 0 values
            comparison.fillna(0, inplace = True)
            a = comparison.iloc[0]
            b = comparison.iloc[1]
            
            # Convert to numpy vectors
            a = a.to_numpy()
            b = b.to_numpy()

            # manually compute cosine similarity
            dot = np.dot(a,b)
            norma = np.linalg.norm(a)
            normb = np.linalg.norm(b)
            cos = round(dot / (norma * normb), 5)
            column.append(cos)
        
        # Insert pair wise comparison as column in final matrix
        insertion = pd.Series(column)
        sims.insert(tick, cols[tick], insertion)

        tick += 1
    
    export_csv = sims.to_csv (r'matrix.csv', index = None, header=True)


    print("--- %s minutes ---" % round(((time.time() - start_time)/60), 2))
    
    

                    
def retrieval(queries, data):
    """
    Function performs a comparison
    analysis between TF-IDF vectorized queries 
    and all documents in the data parameter. Data
    must be in pre-processed format.
    """
    
    start_time = time.time()
    
    # Initialize all tf-idf vectors

    print()
    print("##############")
    print("### PART B ###")
    print("##############")
    print()
    print()
    print('Initializing data...')
    pbar3 = ProgressBar()
    
    start_vectors  = []
    
    # Append each tfd-idf vector to start_vectors (all docs)
    for i in pbar3(range(len(data))):
        start_vectors.append(TFIDFVector(data[i], universe_size(data), document_frequency(data)).vectorize())
    print("TF-IDF vectors computed")
    # Queue up vectors for comparison
    to_compare = queue.Queue(maxsize = len(data))
    cols = ["col" + str(i) for i in range(len(data))]

    for vector in start_vectors:
        to_compare.put(vector)
    
    print('Intializing queries...')
    
    # Pre-process queries
    processed_queries = []
    for query in queries:
        filename = ''
        q = Document(filename, path, text = query)
        processed_queries.append(q.process_query())
    
    
    # Vectorize queries
    query_vectors = []
    for i in range(len(queries)):
        query_vectors.append(TFIDFVector(processed_queries[i], universe_size(data), document_frequency(data)).query_vectorize())
    
    print('Queries processed!')
    
    # Comparisons
    
    
    # Queue up vectors for comparison
    to_search = queue.Queue(maxsize = len(query_vectors))
    cols = ["col" + str(i) for i in range(len(data))]

    for vector in query_vectors:
        to_search.put(vector)
    
    retrievals = pd.DataFrame()
    
    tick = 0
    # Outer loop handles queued vectors
    
    print("Searching...")
    print()
    for vector in query_vectors:
        print()
        print('##########')
        print("Query " + str(tick+1) +":" + "##")
        print('##########')
        print()
        base = to_search.get()
        column_retrievals = []
        
        # Inner loop compares each vector in data to current vector in the queue
        pbar2 = ProgressBar()
        for i in pbar2(range(len(data))):
            
            # Convert pair of vectors to data frame
            comparison = pd.DataFrame([base, start_vectors[i]])
            
            # Replace non-present words from opposing document with 0 values
            comparison.fillna(0, inplace = True)
            a = comparison.iloc[0]
            b = comparison.iloc[1]
            
            # Convert to numpy vectors
            a = a.to_numpy()
            b = b.to_numpy()

            # manually compute cosine similarity
            dot = np.dot(a,b)
            norma = np.linalg.norm(a)
            normb = np.linalg.norm(b)
            cos = round(dot / (norma * normb), 5)
            column_retrievals.append(cos)
        
        # Insert pair wise comparison as column in final matrix
        insertion = pd.Series(column_retrievals)
        retrievals.insert(tick, cols[tick], insertion)
        tick += 1
    print('Done!')
    print("--- %s seconds ---" % round(((time.time() - start_time)), 2))
        
    return retrievals
    
#####################
#### PRE-PROCESS ####
#####################
"""
This section of code combines the above functions
and classes to generate the necessary term counts
and document frequencies to compute individual TF-IDF
vectors for idividual documents. Run to process new 
data. Data is saved to file 'termCounts.txt'.
"""

# Data directory
entries = os.listdir(path)

# Uncomment if file not already created
print()
print("##############")
print("### PART A ###")
print("##############")
print()
time.sleep(1)
# Uncomment for fresh data

if fresh_data:

    results = main(entries)
    print()
print("Pre-processing complete!")


################
### ANALYSIS ###
################

with open('wordCounts.txt', 'r') as file:
    data = file.read()
    data = ast.literal_eval(data)

#################################################################
# Run cosineSim() if not been run already (run time ~ 75 minutes)
if fresh_data:
    matrix = cosineSim(data)
#################################################################

df = pd.read_csv('matrix.csv')
print()
print("##############")
print("### PART C ###")
print("##############")
print()
time.sleep(2)
print(df)



queries = ["""Once upon a time . . . 
there were three little pigs, 
who left their mummy and daddy to see the
world.""",
"""There once lived a poor tailor, who had a son called Aladdin, 
a careless, idle boy who would
do nothing but play all day long 
in the streets with little idle boys like himself."""]




df = retrieval(queries, data)




print()
print("##############")
print("### PART D ###")
print("##############")
print()
time.sleep(1)

df["Document"] = entries

df['query1'] = df["col0"]
df['query2'] = df["col1"]
df.drop(['col0', 'col1'], axis=1)

query1 = pd.DataFrame(df['Document'])
query1['Score'] = df['query1']

query2 = pd.DataFrame(df['Document'])
query2['Score'] = df['query2']

five_d1 = query1.sort_values(by=['Score'], ascending=False).head(n=10)
five_d2 = query2.sort_values(by=['Score'], ascending=False).head(n=10)

print()
print("QUERY 1: ")
print(queries[0])
print()
print(five_d1)
print()
print("QUERY 2: ")
print(queries[1])
print()
print(five_d2)