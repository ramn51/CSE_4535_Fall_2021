from indexer import Indexer
from preprocessor import Preprocessor

from tqdm import tqdm
import heapq
import sys
import random
import json

class BooleanRetrievalRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer() 

    def _merge(self, results=None, daats=True, key_name=None):
        """ Implement the merge algorithm to merge 2 postings list at a time.
            Use appropriate parameters & return types.
            While merging 2 postings list, preserve the maximum tf-idf value of a document.
            To be implemented."""
        merged = {}

        if not daats:
            key = key_name
        else:
            key = 'daatAnd'
            
        for results in results:
            merged.update(results[key])
        
        return merged

    def _daat_and(self, retrieved_postings_list, original_query ,use_skip=False ,use_tf_idf=False):
        """ Implement the DAAT AND algorithm, which merges the postings list of N query terms.
            Use appropriate parameters & return types.
            To be implemented."""
         # log_file = open('daat_log_file.txt', 'w')
        # sys.stdout = log_file
        posting_list_key = 'postingsList' if not use_skip else 'postingsListSkip'
        daat_key = 'daatAnd' if not use_tf_idf else 'daatAndTfIdf'

        starting_ptrs = {term: 0 for term in retrieved_postings_list[posting_list_key]}
        
        ending_ptrs = {term: len(retrieved_postings_list[posting_list_key][term])  for term in retrieved_postings_list[posting_list_key]}

        postings_lists = retrieved_postings_list[posting_list_key]

        # For Merge order optimization
        postings_lists = dict(sorted(postings_lists.items(), key=lambda item: len(item[1])))
        terms = list(postings_lists.keys())

        # Ensure original_query is a string
        if isinstance(original_query, list):
            query_terms_key = ' '.join(original_query)
        else:
            query_terms_key = original_query

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

            # If there is even one term which has reached the end of its postings list then break
            if len(terms_to_remove) > 0:
                break

            for term in terms:
                term_pointer = starting_ptrs[term]
                term_postings_list = postings_lists[term]

                # print("TERM POINTER", term, term_pointer)
                # print("TERM POSTINGS LIST", term, term_postings_list)
                # print("ENDING PTRS", ending_ptrs)
                
                doc_id = term_postings_list[term_pointer][0] if use_tf_idf else term_postings_list[term_pointer]

                if term_pointer < ending_ptrs[term] and doc_id == min_doc_id:
                    # print("INSIDE THE IF of exact match", term_postings_list[term_pointer], min_doc_id)
                    res['daatAnd'][query_terms_key]['num_comparisons'] += 1
                    term_pointer += 1
                    starting_ptrs[term] = term_pointer
                    # print(starting_ptrs)
                    if term_pointer < ending_ptrs[term]:
                        if use_tf_idf:
                            doc_id, tf_idf = term_postings_list[term_pointer]
                            heapq.heappush(min_heap, (doc_id, tf_idf, term))
                        else:
                            doc_id = term_postings_list[term_pointer]
                            heapq.heappush(min_heap, (doc_id, term))
                        # print("MIN HEAP AFTER INSERTION", min_heap)
                    else:
                        docs_matched = False
                    continue
                

                elif term_pointer < ending_ptrs[term] and doc_id > min_doc_id:
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
                while min_heap and (min_doc_id == min_heap[0][0] if use_tf_idf else min_doc_id == min_heap[0]):
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

        # print(res)

        return res
        

    def get_postings(self, query, use_skip=False, get_tf_idf=False):
        """ Function to get the postings list of a term from the index.
            Use appropriate parameters & return types.
            To be implemented."""
        inverted_index_key = 'postingsList' if not use_skip else 'postingsListSkip'
        res = {inverted_index_key: {}}
        inverted_index = self.indexer.get_index()
        for term in query:
            if term not in inverted_index:
                res[inverted_index_key][term] = []
            else:
                if not use_skip:
                    term_docs = inverted_index[term].traverse_list()
                else:
                    term_docs = inverted_index[term].traverse_skips()
                
                res[inverted_index_key][term] = [(doc_id, tf_idf) if get_tf_idf else doc_id for doc_id, term_freq, total_doc_tokens, tf_idf in term_docs]

        return res

    def _output_formatter(self, op):
        """ This formats the result in the required format.
            Do NOT change."""
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_indexer(self, corpus_path):
        """ This function reads & indexes the corpus. After creating the inverted index,
            it sorts the index by the terms, add skip pointers, and calculates the tf-idf scores.
            Already implemented, but you can modify the orchestration, as you seem fit."""
        # with open(corpus, 'r') as fp:
        #     for line in tqdm(fp.readlines()):
        #         doc_id, document = self.preprocessor.get_doc_id(line)
        #         tokenized_document = self.preprocessor.tokenizer(document)
        preprocessed_data = self.preprocessor.preprocess_2(corpus_path)
        inverted_index, total_docs = self.indexer.create_index(preprocessed_data)
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf(total_docs)

    def sanity_checker(self, command):
        """ DO NOT MODIFY THIS. THIS IS USED BY THE GRADER. """

        index = self.indexer.get_index()
        kw = random.choice(list(index.keys()))
        return {"index_type": str(type(index)),
                "indexer_type": str(type(self.indexer)),
                "post_mem": str(index[kw]),
                "post_type": str(type(index[kw])),
                "node_mem": str(index[kw].start_node),
                "node_type": str(type(index[kw].start_node)),
                "node_value": str(index[kw].start_node.value),
                "command_result": eval(command) if "." in command else ""}

    def run_queries(self, query_list, original_query_list, random_command, sanity_checker=False):
        # print(query_list)
        """ DO NOT CHANGE THE output_dict definition"""
        output_dict = {'postingsList': {},
                       'postingsListSkip': {},
                       'daatAnd': {},
                       'daatAndSkip': {},
                       'daatAndTfIdf': {},
                       'daatAndSkipTfIdf': {},
                       'sanity': self.sanity_checker(random_command) if sanity_checker else {}}
        
        daat_results = []
        daat_results_skip = []
        daat_results_tf_idf = []
        daat_results_tf_idf_skip = []

        query_list_tuple = [(query, original_query) for query, original_query in zip(query_list, original_query_list)] 

        # print(query_list_tuple)

        log_file = open('daat_log_file.txt', 'w')
        # sys.stdout = log_file


        postings_lists_set = []
        postings_lists_skip_set = []
        postings_lists_tf_idf_set = []
        postings_lists_tf_idf_skip_set = []

        # print('Query TUPLE', query_list_tuple)

        for query, original_query in tqdm(query_list_tuple, desc="Processing Queries to do DAAT"):
            """ Run each query against the index. You should do the following for each query:
                1. Pre-process & tokenize the query.
                2. For each query token, get the postings list & postings list with skip pointers.
                3. Get the DAAT AND query results & number of comparisons with & without skip pointers.
                4. Get the DAAT AND query results & number of comparisons with & without skip pointers, 
                    along with sorting by tf-idf scores."""
            query_str = " ".join(query)
            postings_list = self.get_postings(query, use_skip=False, get_tf_idf=False)
            postings_list_skip = self.get_postings(query, use_skip=True, get_tf_idf=False)
            postings_list_tf_idf = self.get_postings(query, use_skip=False, get_tf_idf=True)
            postings_list_tf_idf_skip = self.get_postings(query, use_skip=True, get_tf_idf=True)

            postings_lists_set.append(postings_list)
            postings_lists_skip_set.append(postings_list_skip)
            postings_lists_tf_idf_set.append(postings_list_tf_idf)
            postings_lists_tf_idf_skip_set.append(postings_list_tf_idf_skip)

            # print(postings_list_tf_idf_skip, "POSTINGS LIST TF IDF SKIP")
            # print("QUERY", query_str, "POSTINGS LIST", postings_list)
            daat_result_normal = self._daat_and(postings_list, original_query)
            daat_result_skip = self._daat_and(postings_list_skip, original_query, use_skip=True)
            daat_result_tf_idf = self._daat_and(postings_list_tf_idf, original_query, use_tf_idf=True)
            daat_result_tf_idf_skip = self._daat_and(postings_list_tf_idf_skip, original_query, use_skip=True, use_tf_idf=True)

            daat_results.append(daat_result_normal)
            daat_results_skip.append(daat_result_skip)
            daat_results_tf_idf.append(daat_result_tf_idf)
            daat_results_tf_idf_skip.append(daat_result_tf_idf_skip)

        output_dict['daatAnd'] = self._merge(daat_results, daats=True)
        output_dict['daatAndSkip'] = self._merge(daat_results_skip, daats=True)
        output_dict['daatAndTfIdf'] = self._merge(daat_results_tf_idf, daats=True)
        output_dict['daatAndSkipTfIdf'] = self._merge(daat_results_tf_idf_skip, daats=True)

 
        output_dict['postingsList'] = self._merge(postings_lists_set, daats=False, key_name='postingsList')
        output_dict['postingsListSkip'] = self._merge(postings_lists_skip_set, daats=False, key_name='postingsListSkip')

        return output_dict
    


if __name__ == "__main__":
    """ Driver code for the project, which defines the global variables.
        Do NOT change it."""

    output_location = "project2_output.json"
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--corpus", type=str, help="Corpus File name, with path.")
    # parser.add_argument("--output_location", type=str, help="Output file name.", default=output_location)
    # parser.add_argument("--username", type=str,
    #                     help="Your UB username. It's the part of your UB email id before the @buffalo.edu. "
    #                          "DO NOT pass incorrect value here")

    # argv = parser.parse_args()

    # corpus = argv.corpus
    # output_location = argv.output_location
    # username_hash = hashlib.md5(argv.username.encode()).hexdigest()

    """ Initialize the project runner"""
    runner = BooleanRetrievalRunner()

    """ Index the documents from beforehand. When the API endpoint is hit, queries are run against 
        this pre-loaded in memory index. """
    corpus_path = '../project2/data/input_corpus.txt'
    runner.run_indexer(corpus_path)

    queries, original_queries = runner.preprocessor.preprocess_query(file_path='../project2/data/queries.txt')

    print(queries)

    output_dict = runner.run_queries(queries, original_queries ,"runner.indexer.get_index()", sanity_checker=False)
    # print(output_dict)
    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)
    



        