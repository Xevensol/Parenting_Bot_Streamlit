import streamlit as st
from dotenv import load_dotenv
import os
from utils import get_response
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))

index_name = "parenting-bot"

st.title("Parenting Bot Application")
st.write("A parenting bot to answer your queries, handled by Professor Dr. Javed Iqbal.")

def query_bot():
    query = st.text_input("Enter your query:")
    if st.button("Get Response"):
        if query:
            response = get_response(query)
            st.write("Response:")
            st.write(response if response else "No relevant response found.")
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    query_bot()
    