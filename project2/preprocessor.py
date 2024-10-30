'''
@author: Sougata Saha
Institute: University at Buffalo
'''

import collections
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
nltk.download('stopwords')

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def get_doc_id(self, doc):
        """ Splits each line of the document, into doc_id & text.
            Already implemented"""
        arr = doc.split("\t")
        return int(arr[0]), arr[1]

    def tokenizer(self, text):
        """ Implement logic to pre-process & tokenize document text.
            Write the code in such a way that it can be re-used for processing the user's query.
            To be implemented."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        # White Space tokenization
        text = text.strip()
        tokens = text.split()

        invalid_token_status = False

        # Count white space between words
        if any(word == '' or word == ' ' for word in tokens):
            invalid_token_status = True
            raise ValueError("Invalid token found (space or a blank token)")
        
        # Do not include stop words
        tokens = [word for word in tokens if word not in self.stop_words]

        # Performing Porters stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return tokens
    
    def read_file_content(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = file.read()
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def preprocess(self, file_path):
        content = self.read_file_content(file_path=file_path)
        preprocessed_text = []
        postings_dict = {}

        if content[:3]:
            split_content = content.split("\n")
            for line in split_content:
                doc_id, text = self.get_doc_id(line)
                postings_dict[doc_id] = self.tokenizer(text)
                

        print(postings_dict)


