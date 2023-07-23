import rf_iris
text = rf_iris
prompt = f"""
Analyze the python code delimited by triple backticks and answers this question: {question_1}
```{text}```
"""
response = get_completion(prompt)
print(response)