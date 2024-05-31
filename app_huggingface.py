import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, AdamW
from torch.utils.data import DataLoader, RandomSampler
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

def train_model(model, train_dataloader, device):
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.to(device)
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # Assume training is needed and set up a dummy DataLoader
    # You should replace this with actual data loading and preparation
    train_data = {'input_ids': torch.tensor([[0]]), 'attention_mask': torch.tensor([[1]]), 'start_positions': torch.tensor([0]), 'end_positions': torch.tensor([1])}
    train_dataloader = DataLoader([train_data], batch_size=1)
    
    # Check if model needs to be trained and train if necessary
    if st.button("Train Model"):
        train_model(model, train_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
        
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # Load the fine-tuned model for inference
            model = AutoModelForQuestionAnswering.from_pretrained('./fine_tuned_model')
            tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
            qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
            context = ' '.join(chunks)  # This approach may need refinement based on performance
            answer = qa_pipeline(question=user_question, context=context)
            st.write(answer['answer'])

if __name__ == '__main__':
    main()
