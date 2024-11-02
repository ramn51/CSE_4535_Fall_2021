import collections
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import WhitespaceTokenizer
import math
import heapq
# nltk.download('stopwords')

from linkedlist import LinkedList

import sys

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def get_doc_id(self, doc):
        """ Splits each line of the document, into doc_id & text.
            Already implemented"""
        arr = doc.split("\t")
        return int(arr[0]), arr[1]
    
    @staticmethod
    def get_token_freq(token, tokens):
        token_counter = collections.Counter(tokens)
        return token_counter[token]

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

    
    def preprocess_query(self, file_path):
        content = self.read_file_content(file_path=file_path)
        
        if content:
            split_content = content.split("\n")
            queries = []
            for index, line in enumerate(split_content):
                queries.append(self.tokenizer(line))
                
            return queries

    def preprocess(self, file_path):
        content = self.read_file_content(file_path=file_path)

        # Postings dictionary that has k, v pairs of doc_id, tokens
        postings_dict = {}
        docs = []
 
        if content:
            split_content = content.split("\n")
            for line in split_content:
                doc_id, text = self.get_doc_id(line)
                tokens = self.tokenizer(text)
                all_tokens = tokens

                # Only unique tokens, remove duplicates
                tokens = list(set(tokens))

                for token in tokens:
                    if token not in postings_dict:
                        token_freq = get_token_freq(token, all_tokens)
                        postings_dict[token] = LinkedList()
                    postings_dict[token].insert_at_end((doc_id, token_freq, len(all_tokens)))

                docs.append(doc_id)

        total_docs = len(set(docs))

        return postings_dict, total_docs

    def preprocess_2(self, file_path):
        content = self.read_file_content(file_path=file_path)
        preprocessed_data = []

        if content:
            split_content = content.split("\n")
            for line in split_content:
                doc_id, text = self.get_doc_id(line)
                tokens = self.tokenizer(text)
                preprocessed_data.append((doc_id, tokens))

        return preprocessed_data
    
    # Done (verified)
    def create_postings_dict(self, preprocessed_data):
        postings_dict = {}
        docs = []

        for doc_id, tokens in preprocessed_data:
            all_tokens = tokens
            unique_tokens = list(set(tokens))

            for token in unique_tokens:
                if token not in postings_dict:
                    token_freq = get_token_freq(token, all_tokens)
                    postings_dict[token] = LinkedList()
                postings_dict[token].insert_at_end((doc_id, token_freq, len(all_tokens)))

            docs.append(doc_id)

        total_docs = len(set(docs))

        return postings_dict, total_docs
    
    

    # Done (verified)
    def create_skip_postings_list(self, postings_list):
        for k, v in postings_list.items():
            v.add_skip_connections()
        return postings_list
    

    # Done (verified)
    def calculate_tf_idf(self, postings_list, total_docs, use_log=False):
        for k, v in postings_list.items():
            term_docs = v.traverse_list()

            idf = math.log(total_docs / len(term_docs)) if use_log else (total_docs / len(term_docs))

            updated_posting = LinkedList()
            for term_doc in term_docs:
                doc_id, term_freq, total_doc_tokens = term_doc
                tf = term_freq / total_doc_tokens
                tf_idf = tf * idf
                updated_posting.insert_at_end((doc_id, term_freq, total_doc_tokens, tf_idf))
                
            postings_list[k] = updated_posting

        return postings_list
    
    # Done
    def get_postings_list(self, query, postings_list, use_skip=False, get_tf_idf=False):
        res = {'postingsList': {}}
        for term in query:
            if term not in postings_list:
                res['postingsList'][term] = []
            else:
                if not use_skip:
                    term_docs = postings_list[term].traverse_list()
                else:
                    term_docs = postings_list[term].traverse_skips()
                
                res['postingsList'][term] = [(doc_id, tf_idf) if get_tf_idf else doc_id for doc_id, term_freq, total_doc_tokens, tf_idf in term_docs]

        return res
    
    # done (verified)
    def daat_and(self, retrieved_postings_list, use_tf_idf=False):
        # log_file = open('daat_log_file.txt', 'w')
        # sys.stdout = log_file
        
        starting_ptrs = {term: 0 for term in retrieved_postings_list['postingsList']}
        
        ending_ptrs = {term: len(retrieved_postings_list['postingsList'][term])  for term in retrieved_postings_list['postingsList']}

        postings_lists = retrieved_postings_list['postingsList']
        # For Merge order optimization
        postings_lists = dict(sorted(postings_lists.items(), key=lambda item: len(item[1])))
        terms = list(postings_lists.keys())

        query_terms_key = ' '.join(f'{term}' for term in terms)
        res = {'daatAnd': {query_terms_key: {'results': [], 'num_comparisons': 0, 'num_docs': 0}}}


        # Add the first element of each postings list to the min heap
        # If its tf-idf ordering then add the doc_id still but the tuple needs to destructured
        if use_tf_idf:
            min_heap = [(postings_lists[term][0][0], postings_lists[term][0][1], term) for term in terms]
        else:
            min_heap = [(postings_lists[term][0], term) for term in terms]
        
        heapq.heapify(min_heap)

        # print("MIN HEAP INITAIL", min_heap)

        while min_heap:
            if use_tf_idf:
                min_doc_id, min_tf_idf, min_term = heapq.heappop(min_heap)
            else:
                min_doc_id, min_term = heapq.heappop(min_heap)

            docs_matched = True

            # print("MIN DOC ID", min_doc_id)
            # print("heap after pop", min_heap)

            # Take out the term which dont need to iterated as its length is traversed
            terms_to_remove = []

            for term, start_ptr in starting_ptrs.items():
                for term_, end_ptr in ending_ptrs.items():
                    if term == term_ and start_ptr == end_ptr:
                        terms_to_remove.append(term)

            # print("TERMS TO REMOVE", terms_to_remove)
            
            terms = [i for i in terms if i not in terms_to_remove]
            # print('TERMS UPDATED', terms)

            # If there is only one term then dont have to continue since it will not result in an AND match
            if len(terms) == 1:
                break

            for term in terms:
                term_pointer = starting_ptrs[term]
                term_postings_list = postings_lists[term]

                # print("TERM POINTER", term, term_pointer)
                # print("TERM POSTINGS LIST", term, term_postings_list)
                # print("ENDING PTRS", ending_ptrs)
                

                if term_pointer < ending_ptrs[term] and term_postings_list[term_pointer] == min_doc_id:
                    # print("INSIDE THE IF of exact match", term_postings_list[term_pointer], min_doc_id)
                    res['daatAnd'][query_terms_key]['num_comparisons'] += 1
                    term_pointer += 1
                    starting_ptrs[term] = term_pointer
                    # print(starting_ptrs)
                    if term_pointer < ending_ptrs[term]:
                        heapq.heappush(min_heap, (term_postings_list[term_pointer], term))
                        # print("MIN HEAP AFTER INSERTION", min_heap)
                    else:
                        docs_matched = False
                    continue
                

                elif term_pointer < ending_ptrs[term] and term_postings_list[term_pointer] > min_doc_id:
                    res['daatAnd'][query_terms_key]['num_comparisons'] += 1
                    # print("INSIDE THE ELIF of not exact match", term_postings_list[term_pointer], min_doc_id)
                    docs_matched = False

                elif term_pointer >= ending_ptrs[term]:
                    # print('ENDING TERM PTR REACHED FOGR TERM', term)
                    continue


            # print("STARTING PTRS", starting_ptrs, "DOC MATCHED", docs_matched)
            if docs_matched:
                if use_tf_idf:
                    res['daatAnd'][query_terms_key]['results'].append((min_doc_id, min_tf_idf))
                else:
                    res['daatAnd'][query_terms_key]['results'].append(min_doc_id)
                res['daatAnd'][query_terms_key]['num_docs'] += 1
                # for term in terms:
                #     starting_ptrs[term] += 1
                # print('MIN DOC ID BEFORE THE CLEARING LOOP', min_doc_id, min_heap, starting_ptrs)
                while min_heap and min_doc_id == min_heap[0][0]:
                    if use_tf_idf:
                        min_doc_id, min_tf_idf, min_term = heapq.heappop(min_heap)
                    else:
                        min_doc_id, min_term = heapq.heappop(min_heap)
                    # print('Poping from heap', min_doc_id, min_heap)

                # print('MIN DOC ID AFTER THE CLEARING LOOP', min_doc_id, min_heap, starting_ptrs)

            # print("RES AFTER THE LOOP", res)


        

        if use_tf_idf:
            res['daatAnd'][query_terms_key]['results'].sort(key=lambda x: x[1], reverse=True)
            # Remove the tf-idf values from the results
            res['daatAnd'][query_terms_key]['results'] = [doc_id for doc_id, _ in res['daatAnd'][query_terms_key]['results']]

        print(res)

        return res
    
    # done (verified)
    def merge_daat_results(self, daat_results):
        merged = {'daatAnd': {}}

        for results in daat_results:
            merged['daatAnd'].update(results['daatAnd'])
        
        return merged
            

    

p = Preprocessor()
preprocessed_postings_dict, total_docs = p.preprocess('../project2/data/input_corpus.txt')


norm_items = preprocessed_postings_dict['effect'].traverse_list()
print(norm_items, len(norm_items))

print("Using the seperated methods to preprocess and create postings list")
preprocessed_data = p.preprocess_2('../project2/data/input_corpus.txt')
postings_dict, total_docs = p.create_postings_dict(preprocessed_data)

processed_items = postings_dict['effect'].traverse_list()

print("POSTINGS DICT", processed_items, len(processed_items))


skip_postings_list_dict = p.create_skip_postings_list(preprocessed_postings_dict)


tf_idf_postings_dict = p.calculate_tf_idf(preprocessed_postings_dict, total_docs)
tf_idf_items = tf_idf_postings_dict['effect'].traverse_list()
print('TF IDF List\n',tf_idf_items, len(tf_idf_items))

skip_postings_list_dict = p.create_skip_postings_list(tf_idf_postings_dict)

skip_items = tf_idf_postings_dict['effect'].traverse_skips()
print('TF IDF Skip LIST\n', skip_items, len(skip_items))

# print(tf_idf_postings_dict['covid19'].traverse_list())

# print(p.preprocess_query('../project2/data/queries.txt'))

queries = p.preprocess_query('../project2/data/queries.txt')



# test_query = [['novel', 'coronaviru']]
test_query = [['hydroxychloroquin', 'effect']]
for query in queries:
    print(query)
    # result_no_skip = p.get_postings_list(query, tf_idf_postings_dict, use_skip=False)
    result_with_skip = p.get_postings_list(query, tf_idf_postings_dict, use_skip=False)

    # print("Without skip:", result_no_skip)
    print("With skip:", result_with_skip)

test_query2 = [['apple', 'banana', 'cherry']]

test_postings_list = {
    'postingsList': {
        'apple': [1, 4, 5, 8, 10, 14],
        'banana': [2, 4, 8, 10, 12, 15],
        'cherry': [4, 8, 9, 10, 13]
    }
}

# print(test_postings_list)

# for query in test_query2:
#     p.daat_and(test_postings_list)
print(queries)
# print(p.get_postings_list(query, tf_idf_postings_dict, use_skip=False))

print("WITHOUT USING SKIP\n")
# Merge all results into a single dict for all queries
daat_results = []
for query in queries:
    daat_results.append(p.daat_and(p.get_postings_list(query, tf_idf_postings_dict, use_skip=False, get_tf_idf=False)))

print(p.merge_daat_results(daat_results))

# print("USING SKIP\n")
# for query in queries:
#     p.daat_and(p.get_postings_list(query, tf_idf_postings_dict, use_skip=True, get_tf_idf=True))

# Build sorted TF_IDF postings list


