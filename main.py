#Integrate our code OpenAI API
import os
import openai
from constants import openai_key
from langchain_openai import OpenAI

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

st.title("Langchain Demo with OpenAI")

input_text=st.text_input("Search the topic you want ")

llm = OpenAI(temperature=0.7)


if input_text:
    st.write(llm.invoke(input_text))


    
    
