'''
@author: Sougata Saha
Institute: University at Buffalo
'''

from linkedlist import LinkedList
from collections import OrderedDict
from preprocessor import Preprocessor
import math

class Indexer:
    def __init__(self):
        """ Add more attributes if needed"""
        self.inverted_index = OrderedDict({})

    def get_index(self, skip_connections=False):
        """ Function to get the index.
            Already implemented."""
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        """ This function adds each tokenized document to the index. This in turn uses the function add_to_index
            Already implemented."""
        doc_length = len(tokenized_document)
        unique_tokens = list(set(tokenized_document))
        for token in unique_tokens:
            token_freq = Preprocessor.get_token_freq(token, tokenized_document)
            self.add_to_index(token, doc_id, token_freq, doc_length)


    def add_to_index(self, term, doc_id, token_freq, doc_length):
        """ This function adds each term & document id to the index.
            If a term is not present in the index, then add the term to the index & initialize a new postings list (linked list).
            If a term is present, then add the document to the appropriate position in the posstings list of the term.
            To be implemented."""
        if term not in self.inverted_index:
            self.inverted_index[term] = LinkedList()
        self.inverted_index[term].insert_at_end((doc_id, token_freq, doc_length))

        
    def create_index(self, preprocessed_data):
        docs = []

        for doc_id, tokens in preprocessed_data:
            self.generate_inverted_index(doc_id, tokens)
            docs.append(doc_id)

        total_docs = len(docs)
        return self.inverted_index, total_docs

    def create_postings_dict(self, preprocessed_data):
        postings_dict = {}
        docs = []

        for doc_id, tokens in preprocessed_data:
            all_tokens = tokens
            unique_tokens = list(set(tokens))

            for token in unique_tokens:
                if token not in postings_dict:
                    token_freq = Preprocessor.get_token_freq(token, all_tokens)
                    postings_dict[token] = LinkedList()
                postings_dict[token].insert_at_end((doc_id, token_freq, len(all_tokens)))

            docs.append(doc_id)

        total_docs = len(set(docs))

        return postings_dict, total_docs

    def sort_terms(self):
        """ Sorting the index by terms.
            Already implemented."""
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        """ For each postings list in the index, add skip pointers.
            To be implemented."""
        for k, v in self.inverted_index.items():
            v.add_skip_connections()
        return self.inverted_index
    
    def get_postings_list(self, query, use_skip=False, get_tf_idf=False):
        inverted_index_key = 'postingsList' if not use_skip else 'postingsListSkip'
        res = {inverted_index_key: {}}
        for term in query:
            if term not in self.inverted_index:
                res[inverted_index_key][term] = []
            else:
                if not use_skip:
                    term_docs = self.inverted_index[term].traverse_list()
                else:
                    term_docs = self.inverted_index[term].traverse_skips()
                
                res[inverted_index_key][term] = [(doc_id, tf_idf) if get_tf_idf else doc_id for doc_id, term_freq, total_doc_tokens, tf_idf in term_docs]

        return res
    

    def calculate_tf_idf(self, total_docs, use_log=False):
        """ Calculate tf-idf score for each document in the postings lists of the index.
            To be implemented."""
        updated_index = OrderedDict({})

        for term, posting_list in self.inverted_index.items():
            term_docs = posting_list.traverse_list()

            idf = math.log(total_docs / len(term_docs)) if use_log else (total_docs / len(term_docs))

            updated_posting = LinkedList()
            for term_doc in term_docs:
                doc_id, term_freq, total_doc_tokens = term_doc
                tf = term_freq / total_doc_tokens
                tf_idf = tf * idf
                updated_posting.insert_at_end((doc_id, term_freq, total_doc_tokens, tf_idf))

            if hasattr(posting_list, 'skip_length'):
                updated_posting.skip_length = posting_list.skip_length
                updated_posting.n_skips = posting_list.n_skips
                updated_posting.add_skip_connections()
                
            updated_index[term] = updated_posting

        self.inverted_index = updated_index

        return self.inverted_index
    


if __name__ == '__main__':
    p = Preprocessor()
    preprocessed_data = p.preprocess_2('../project2/data/input_corpus.txt')
    Indexer1 = Indexer()
    Indexer2 = Indexer()
    inverted_index, total_docs =  Indexer1.create_index(preprocessed_data)

    print("WITHOUT ADDING SKIP (NEW WAY WITH SEPARATION)", inverted_index['effect'].traverse_list(), len(inverted_index['effect'].traverse_list()))

    inverted_index_norm, total_docs_norm =  Indexer1.create_postings_dict(preprocessed_data)
    print("WITHOUT ADDING SKIP OLD WAY (USING THE METHOD DOES FULLY)", inverted_index_norm['effect'].traverse_list())

    # Adding skip pointers
    inverted_index2, total_docs2 =  Indexer2.create_index(preprocessed_data)
    inverted_index2 = Indexer2.add_skip_connections()
    print("WITH ADDING SKIP", inverted_index2['effect'].traverse_skips())

    tf_idf_index = Indexer2.calculate_tf_idf(total_docs)
    # print(tf_idf_index['effect'].traverse_list())
    print("WITH ADDING SKIP AND CALCULATING TF-IDF", tf_idf_index['effect'].traverse_skips())
