#!/usr/bin/env python3

from src.app.injestion import load_json_data
import os

data_file = os.path.join('data', 'data-2.json')
docs = load_json_data(data_file)
print(f'Loaded {len(docs)} documents')

if docs:
    first_doc = docs[0]
    print(f'First doc keys: {list(first_doc.keys())}')
    print(f'First doc user_question: {first_doc.get("user_question", "N/A")[:100]}...')
    print(f'First doc answer: {first_doc.get("answer", "N/A")[:100]}...')
    print(f'Follow-up questions structure: {type(first_doc.get("follow_up_questions", []))}')
    
    if first_doc.get('follow_up_questions'):
        print(f'Number of follow-ups: {len(first_doc["follow_up_questions"])}')
        print(f'First follow-up: {first_doc["follow_up_questions"][0]}')
else:
    print('No docs loaded')