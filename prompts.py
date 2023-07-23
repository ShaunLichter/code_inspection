import rf_iris
text = rf_iris
prompt = f"""
Analyze the python code delimited by triple backticks and answers this question: {question_1}
```{text}```
"""
response = get_completion(prompt)
print(response)


messages =  [
{'role':'system',
 'content':"""You are an assistant """},
{'role':'user',
 'content':"""write me a python random forest model for the iris dataset"""},
]
from open_ai_helps import get_completion_from_messages, get_completion_and_token_count
response = get_completion_from_messages(messages, temperature=1)
print(response)


response, token_dict = get_completion_and_token_count(messages)