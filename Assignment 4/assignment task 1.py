import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

nltk.download('stopwords')
nltk.download('punkt')

# Task 1 a Define the text 
text = "Intelligent behavior in people is a product of the mind. But the mind itself is more like what the human brain does."

# Task 1 a  Tokenize the text
words = word_tokenize(text)

# Task 1 a  Remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]

# Task 1 a  Create an inverted index
inverted_index = defaultdict(list)

for position, term in enumerate(filtered_words):
    inverted_index[term].append(position)


print("Task 1 a:")
# Task 1 a  Print the inverted index
for term, positions in inverted_index.items():
    print(f'{term}: {positions}')


# ASSUMPTION TASK 1 b: We use the filtered words from Task 1 a as input for Task 1 b

# Task 1 b Define a function that assigns each word to a block
def blockify(filtered_words, block_size):
    words_to_block = []
    block_number = 0
    for term in filtered_words:
        if(len(words_to_block) % block_size == 0 ):
            block_number = block_number + 1
        words_to_block.append((term, block_number))
    return words_to_block

print("Task 1 b:")
# Task 1 b Print the dictionary that assigns each word to a block
words_to_block = blockify(filtered_words, 4)
for word, block_number in words_to_block:
    print(f'{word} is in Block {block_number}')
    

class Node:
    def __init__(self):
        self.children = {}

def add_word_to_suffix_tree(root, word):
    current_node = root
    for char in word:
        if char not in current_node.children:
            current_node.children[char] = Node()
        current_node = current_node.children[char]

def build_partial_vocabulary_suffix_tree(text):
    root = Node()
    words = text.split()
    for word in words:
        add_word_to_suffix_tree(root, word)

    return root

# Build the partial vocabulary suffix tree
root = build_partial_vocabulary_suffix_tree(text)

# Function to print the suffix tree
def print_suffix_tree(node, word=""):
    if not node.children:
        print(word)
        return
    
    for char, child in node.children.items():
        print_suffix_tree(child, word + char)

print("Task 1 c:")
print_suffix_tree(root)

# Task d Define the documents
documents = [
    "Although we know much more about the human brain than we did even",
    "ten years ago, the thinking it engages in remains pretty much a total",
    "mystery. It is like a big jigsaw puzzle where we can see many of the",
    "pieces, but cannot yet put them together. There is so much about us",
    "that we do not understand at all."
]

# Task d Create a list to store wordlists for each document
wordlists = []

# Task d Process each document to create wordlists without stop words
for document in documents:
    words = word_tokenize(document)
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalpha()]
    wordlists.append(filtered_words)

# Task d Print wordlists for each document
print("Task 1 d step 1:")
for i, wordlist in enumerate(wordlists, 1):
    print(f'D{i} Wordlist: {wordlist}')

# Task d  Step 2: Construct posting lists
posting_lists = {}

# Task d  Iterate through each document and create posting lists
for i, wordlist in enumerate(wordlists, 1):
    for word in wordlist:
        if word in posting_lists:
            posting_lists[word].append(f'D{i}')
        else:
            posting_lists[word] = [f'D{i}']

print("Task 1 d step 2:")
# Task d  Print posting lists for each word
for word, postings in posting_lists.items():
    print(f'{word}: {", ".join(postings)}')