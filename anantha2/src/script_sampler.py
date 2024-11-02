import json
from preprocessor import Preprocessor

file_name = '../project2/data/sample_output.json'
local_file_verifier = './project2_output.json'

file_name_to_verify = './output_flask.json'


fields_to_verify = ['postingsList', 'postingsListSkip']


original_output_fields = []


with open(file_name, 'r') as file:
    original_data = json.load(file)['Response']
    original_output_fields.append(original_data.keys())

print(original_output_fields)

with open(file_name_to_verify, 'r') as file:
        verifier_data = json.load(file)['Response']
        
        # verifier_data = json.load(file)
        

p = Preprocessor()
query_terms = p.preprocess_query(file_path='queries.txt')

print(query_terms)

query_field_tuple_list = [(query, field) for query in query_terms[0] for field in fields_to_verify]

print(query_field_tuple_list)

for query ,field_to_verify in query_field_tuple_list:
    for term in query:
        original_value = original_data[field_to_verify][term]
        verifier_value = verifier_data[field_to_verify][term]

        # Print or perform your verification
        # print(f"Original: {original_value}, Length: {len(original_value)} for Term: {term} for Key {field_to_verify}")
        # print(f"Verifier: {verifier_value}, Length: {len(verifier_value)} for Term: {term} for Key {field_to_verify}")
        print(f"Are they equal: {original_value == verifier_value}")
        if original_value != verifier_value:
            print(f"Original: {original_value}, Length: {len(original_value)} for Term: {term} for Key {field_to_verify}")
            print(f"Verifier: {verifier_value}, Length: {len(verifier_value)} for Term: {term} for Key {field_to_verify}")


# Daat Verification
original_queries = query_terms[1]

fields_to_verify = ['daatAnd', 'daatAndSkip', 'daatAndSkipTfIdf', 'daatAndTfIdf']

query_field_tuple_list = [(query, field) for query in original_queries for field in fields_to_verify]

validation_field = 'results'
num_docs_field = 'num_docs'

true_count = 0
false_count = 0

false_cases = []

for query, field_to_verify in query_field_tuple_list:
    print(f"Query: {query}, Field: {field_to_verify}")
    original_value = original_data[field_to_verify][query]
    verifier_value = verifier_data[field_to_verify][query]

        # Print or perform your verification
    print(f"Original: {original_value}, Length: {len(original_value)} for Term: {term} for Key {field_to_verify}")
    print(f"Verifier: {verifier_value}, Length: {len(verifier_value)} for Term: {term} for Key {field_to_verify}")

    # True count and False count

    if original_value[validation_field] == verifier_value[validation_field] and original_value[num_docs_field] == verifier_value[num_docs_field]:
        true_count += 1
    else:
        false_count += 1
        false_cases.append((query, field_to_verify, original_value, verifier_value))
    
    print(f"Are they equal: {original_value[validation_field] == verifier_value[validation_field]}")
        
print(f"True Count: {true_count}, False Count: {false_count}")

print("FALSE CASES", false_cases)
