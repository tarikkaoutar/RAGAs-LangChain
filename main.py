
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

def main():

    st.set_page_config(page_title="Ask your PDF")

    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.header("Ask your PDF 💬")
    with st.sidebar:
        st.markdown('''
        Upload pdf and ask question related to content in pdf

        ## - Gao Dalie

        - [Youtube](https://www.youtube.com/@GaoDalie_A)
        
        ''')
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      vectorstore = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        retriever = vectorstore.as_retriever()
        
        # Define LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Define prompt template
        template = """You are a helpful, respectful and honest assistant. 
        Always answer as helpfully as possible, while being safe. 
        Your answers should not include any harmful, unethical, racist, sexist, toxic,
        dangerous, or illegal content. Please ensure that your responses are socially 
        unbiased and positive in nature.
        If a question does not make any sense, or is not factually coherent, 
        explain why instead of answering something not correct.
        If you don't know the answer to a question, please don't share false information.
        Question: {question} 
        Context: {context} 
        Answer:
        """


        prompt = ChatPromptTemplate.from_template(template)

        # Setup RAG pipeline
        rag_chain = (
            {"context": retriever,  "question": RunnablePassthrough()} 
            | prompt 
            | llm
            | StrOutputParser() 
        )

        questions = ["What is Software Engineering ", 
             "What is the Phases of Agile model",
            ]
        
        ground_truths = [["Software Engineering is a framework for building software and is an engineering approach to software development"],
                        ["Requirements gathering , Design the requirements , Construction/ iteration , Testing/ Quality assurance , Deployment ,Feedback"],
                    ]
        
        answers = []
        contexts = []

        # Inference
        for query in questions:
            answers.append(rag_chain.invoke(query))
            contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

        # To dict
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths
        }

        # Convert dict to dataset
        dataset = Dataset.from_dict(data)
        result = evaluate(
        dataset = dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        )
        
        df = result.to_pandas()
        st.write(df)

if __name__ == '__main__':
    main()

