import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

st.title("Celebrity Search Results")

input_text = st.text_input("Enter the celebrity name you want to search")

llm = OpenAI(temperature=0.7)

# Prompt 1
first_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about the celebrity named {name}."
)

# Prompt 2
second_prompt = PromptTemplate(
    input_variables=["person"],
    template="From the following text, tell me when the person was born:\n{person}, Dont give extra text just give the year"
)

# Prompt 3
third_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Give me 3 major events that happened on {dob}."
)

# Individual chains
chain1 = first_prompt | llm
chain2 = second_prompt | llm
chain3 = third_prompt | llm


if input_text:
    bio = chain1.invoke({"name": input_text})
    dob = chain2.invoke({"person": bio})
    events = chain3.invoke({"dob": dob})

    st.subheader("Biography")
    st.write(bio)

    st.subheader("Date of Birth")
    st.write(dob)

    st.subheader("Major Events")
    st.write(events)
