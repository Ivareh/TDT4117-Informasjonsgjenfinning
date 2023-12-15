import random # Task 1.0
import codecs
import re
import string
from nltk import FreqDist
import gensim.corpora as corpora
import gensim.models as models
import gensim.similarities as similarities
from nltk.stem import PorterStemmer

random.seed(123)

# Open the file in utf-8 mode task 1.1
with codecs.open("pg3300.txt", "r", "utf-8") as f:
    data = f.read()

# Initialize the Porter Stemmer
stemmer = PorterStemmer() # Task 1.6

# Initialize a FreqDist object to count word frequencies
freqDist = FreqDist()

# Task 1.2 Split the data into paragraphs based on empty lines
original_paragraphs = re.split(r'\n\s*\n', data)

# Make a copy of original paragraphs to keep them for later use
copied_original_paragraphs = list(original_paragraphs)

# Task 1.3 Filter out paragraphs containing the word "Gutenberg"
filtered_paragraphs = [paragraph for paragraph in original_paragraphs if "Gutenberg" not in paragraph]

# Define the list of stopwords
stop_words = [
    "'tis", "'twas", "a", "able", "about", "across", "after", "ain't", "all", "almost",
    "also", "am", "among", "an", "and", "any", "are", "aren't", "as", "at", "be", "because",
    "been", "but", "by", "can", "can't", "cannot", "could", "could've", "couldn't", "dear",
    "did", "didn't", "do", "does", "doesn't", "don't", "either", "else", "ever", "every",
    "for", "from", "get", "got", "had", "has", "hasn't", "have", "he", "he'd", "he'll",
    "he's", "her", "hers", "him", "his", "how", "how'd", "how'll", "how's", "however", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "just", "least", "let", "like", "likely", "may", "me", "might", "might've", "mightn't",
    "most", "must", "must've", "mustn't", "my", "neither", "no", "nor", "not", "of", "off",
    "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says",
    "shan't", "she", "she'd", "she'll", "she's", "should", "should've", "shouldn't", "since",
    "so", "some", "than", "that", "that'll", "that's", "the", "their", "them", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "tis", "to", "too", "twas", "us", "wants", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "were", "weren't", "what", "what'd", "what's", "when", "when", "when'd", "when'll", "when's",
    "where", "where'd", "where'll", "where's", "which", "while", "who", "who'd", "who'll",
    "who's", "whom", "why", "why'd", "why'll", "why's", "will", "with", "won't", "would",
    "would've", "wouldn't", "yet", "you", "you'd", "you'll", "you're", "you've", "your"
]


# Tokenize paragraphs (split them into words) and preprocess them
def preprocess_paragraph(paragraph):
    # Remove punctuation and white characters, convert to lowercase
    paragraph = paragraph.translate(str.maketrans('', '', string.punctuation)) # 1.5 remove punctuation
    paragraph = paragraph.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') # 1.5 replace white characters
    # Split into words
    words = paragraph.split() # Tsak 1.4
    # Stem the words using Porter Stemmer and update the FreqDist
    stemmed_words = [stemmer.stem(word.lower()) for word in words] # Task 1.6
    freqDist.update(stemmed_words)
    return stemmed_words


# Preprocess and tokenize paragraphs
tokenized_paragraphs = [preprocess_paragraph(paragraph) for paragraph in filtered_paragraphs]

# Example: To see how many times the word "tax" appears in the text
print("Tax appears in text: " + freqDist["tax"].__str__() + " times \n") # Task 1.7

# Task 2.1 Build a dictionary using Gensim
dictionary = corpora.Dictionary(tokenized_paragraphs)

# Task 2.1 Filter out stopwords using the stopword list
stop_ids = [dictionary.token2id[word] for word in stop_words if word in dictionary.token2id] # Filters out stopwords
dictionary.filter_tokens(bad_ids=stop_ids) # Filters out stopwords

# Task 2.2 Map paragraphs into Bags-of-Words using the dictionary
corpus = [dictionary.doc2bow(paragraph) for paragraph in tokenized_paragraphs]

# Task 3.1 Build a TF-IDF model using the corpus
tfidf_model = models.TfidfModel(corpus)

# Task 3.2 Map Bags-of-Words into TF-IDF weights
tfidf_corpus = tfidf_model[corpus]

# Task 3.3 Construct a MatrixSimilarity object for calculating similarities with TF-IDF weights
tfidf_similarity = similarities.MatrixSimilarity(tfidf_corpus)

# Task 3.4 - Create an LSI model with 100 topics
lsi_model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[tfidf_corpus] # Transform the corpus using the LSI mode
lsi_similarity = similarities.MatrixSimilarity(lsi_corpus) # Construct a MatrixSimilarity object for calculating similarities with LSI model

# Task 3.5 Report and try to interpret 3 LSI topics
print(lsi_model.show_topics(num_topics=3))
print("\n")
# My interpretation from the topics:
print()

# Task 4.1 Define the query and preprocess it
query = "What is the function of money?"
query = preprocess_paragraph(query)
query_bow = dictionary.doc2bow(query) # Task 4.1 Convert the query into a BOW representation using the same dictionary

# Task 4.2 Convert the BOW representation of the query to TF-IDF representation
query_tfidf = tfidf_model[query_bow]

# Task 4.3 Calculate similarities between the query and paragraphs using TF-IDF model
similarity_scores = list(enumerate(tfidf_similarity[query_tfidf]))

# Task 4.3 Sort and get the top 3 most relevant paragraphs
top_paragraphs = sorted(similarity_scores, key=lambda x: -x[1])[:3]

# Task 4.3 Display the top 3 most relevant paragraphs in their original form (truncated to the first 5 lines)
for paragraph_index, similarity_score in top_paragraphs:
    print(f"Top Relevant Paragraph (Original Form) - Paragraph {paragraph_index + 1}:")
    original_paragraph = copied_original_paragraphs[paragraph_index]
    # Truncate to the first 5 lines
    truncated_paragraph = "\n".join(original_paragraph.splitlines()[:5])
    print(truncated_paragraph)
    print(f"Similarity Score: {similarity_score:.2f}\n")


# Task 4.4 Convert the TF-IDF representation of the query to LSI representation
query_lsi = lsi_model[query_tfidf]

# Task 4.4 Get the top 3 significant LSI topics with the most significant weights
top_lsi_topics = sorted(query_lsi, key=lambda kv: -abs(kv[1]))[:3]

# Task 4.4 Print the top 3 LSI topics and their weights
print("Top 3 LSI Topics for the Query:")
for index, weight in top_lsi_topics:
    print(f"Topic {index}: Weight = {weight:.4f}")
    print(lsi_model.show_topic(index))

# Task 4.4 Calculate similarities between the LSI query and paragraphs using LSI model
lsi_similarity_scores = list(enumerate(lsi_similarity[query_lsi]))

#Task 4.4  Sort and get the top 3 most relevant paragraphs with LSI
top_lsi_paragraphs = sorted(lsi_similarity_scores, key=lambda x: -x[1])[:3]

# Task 4.4 Display the top 3 most relevant paragraphs using LSI in their original form (truncated to the first 5 lines)
print("\nTop 3 Most Relevant Paragraphs according to LSI Model:")
for paragraph_index, similarity_score in top_lsi_paragraphs:
    print(f"Paragraph {paragraph_index + 1}:")
    original_paragraph = copied_original_paragraphs[paragraph_index]
    # Truncate to the first 5 lines
    truncated_paragraph = "\n".join(original_paragraph.splitlines()[:5])
    print(truncated_paragraph)
    print(f"LSI Similarity Score: {similarity_score:.4f}\n")

# Task 4.4 Compare retrieved paragraphs with the paragraphs found for the TF-IDF model
print("\nComparison with TF-IDF Model:")
for i in range(3):
    tfidf_paragraph_index = top_paragraphs[i][0]
    lsi_paragraph_index = top_lsi_paragraphs[i][0]
    print(f"Top TF-IDF Paragraph {i + 1}:")
    print(copied_original_paragraphs[tfidf_paragraph_index].splitlines()[0])
    print("Top LSI Paragraph {i + 1}:")
    print(copied_original_paragraphs[lsi_paragraph_index].splitlines()[0])
    print()
