from transformers import pipeline

# Initialize the Hugging Face pipeline for question-answering with a pre-trained model
qa_pipeline = pipeline('question-answering', model='bert-base-uncased')

def answer_question(question, context):
    return qa_pipeline({'question': question, 'context': context})

# Example usage
context = "Paris is the capital and most populous city of France."
question = "What is the capital of France?"
result = answer_question(question, context)
print(result['answer'])  # Outputs: 'Paris'
