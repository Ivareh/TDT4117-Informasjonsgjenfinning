import string
from elasticsearch import Elasticsearch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import os


ELASTIC_PASSWORD = "Arceus1509"

# Initialize Elasticsearch connection
es = Elasticsearch([{'host': 'localhost', 'port':9200, 'scheme':'http'}], basic_auth=('ivarsuper', "Arceus1509"))

stop_words = set(stopwords.words('english'))

# Step 2: Document Preprocessing
def preprocess_text(text):
    # Remove punctuationt
    tokens = ["".join(char for char in text if char not in string.punctuation).lower()]
    # Remove stop words (You can define your own list of stop words)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Define the directory where the text files are located
directory = "DataAssignment4/"

# List of file names
file_names = ["Text1.txt", "Text2.txt", "Text3.txt", "Text4.txt"]

# Initialize an empty list to store the combined content
combined_content = []

# Loop through the file names and read their contents
for file_name in file_names:
    file_path = os.path.join(directory, file_name)
    
    # Check if the file exists
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            file_contents = file.read()
            combined_content.append(file_contents)
    else:
        print(f"File {file_name} does not exist in the specified directory.")



# Preprocess the combined content
preprocessed_content = [preprocess_text(text) for text in combined_content]


def tokenize(text):
    words = text[0].split()
    return words

# Task 1 a  Create an inverted index
inverted_index = defaultdict(list)

for realposition, text in enumerate(preprocessed_content):
    text = tokenize(text)
    for position, term in enumerate(text):
        inverted_index[term].append(realposition)

for doc_id, document in enumerate(combined_content):
    es.index(index="index_name", id=doc_id, body=document)

# Perform a simple search
search_query = {
    "query": {
        "match": {"content": "claims the duty"}
    }
}

search_result = es.search(index="index_name", body=search_query)
print(search_result)