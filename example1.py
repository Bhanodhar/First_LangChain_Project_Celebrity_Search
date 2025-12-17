#Integrate our code OpenAI API
import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

st.title("Celebrity Search Results")

input_text=st.text_input("Enter the celebrity name you want to search ")


# Define a prompt template
first_prompt= PromptTemplate(
    input_variables=['name'],
    template="Tell me about the celebrity named {name}."
)

second_prompt = PromptTemplate(
    input_variables=["person"],
    template="When was {person} born?"
)

third_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Give me 3 major events happened on {dob}?"
)

llm = OpenAI(temperature=0.7)

chain = first_prompt | llm | second_prompt | llm | third_prompt | llm   # modern runnable chain


if input_text:
    st.write(chain.invoke({"name": input_text}))



