### Memory Sharing Example with Multiple Chains

import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

st.title("Celebrity Search Results with Memory")

input_text = st.text_input("Enter the celebrity name")

llm = OpenAI(temperature=0.7)

# Shared memory
memory = ConversationBufferMemory(return_messages=True)

# Prompt 1
prompt1 = PromptTemplate(
    input_variables=["name"],
    template="Tell me about the celebrity named {name}."
)

# Prompt 2
prompt2 = PromptTemplate(
    input_variables=["text"],
    template="From the following text, tell me when the person was born:\n{text}"
)

# Prompt 3
prompt3 = PromptTemplate(
    input_variables=["dob"],
    template="Give me 3 major world events that happened on {dob}."
)

# Chains with shared memory
chain1 = prompt1 | llm
chain2 = prompt2 | llm
chain3 = prompt3 | llm

if input_text:
    def to_text(value):
        # Normalize different possible return types to a plain string
        try:
            if value is None:
                return ""
            # BaseMessage-like objects have a `content` attribute
            content = getattr(value, "content", None)
            if isinstance(content, str):
                return content
            # dict-like results
            if isinstance(value, dict):
                for k in ("text", "result", "output", "content"):
                    if k in value and isinstance(value[k], str):
                        return value[k]
                # fallback to joining stringifiable values
                return " ".join(str(v) for v in value.values())
            # lists
            if isinstance(value, (list, tuple)):
                return "\n".join(to_text(v) for v in value)
            # fallback to str()
            return str(value)
        except Exception:
            return str(value)

    bio = chain1.invoke({"name": input_text})
    bio_text = to_text(bio)
    memory.save_context({"name": input_text}, {"bio": bio_text})

    dob = chain2.invoke({"text": bio_text})
    dob_text = to_text(dob)
    memory.save_context({"text": bio_text}, {"dob": dob_text})

    events = chain3.invoke({"dob": dob_text})
    events_text = to_text(events)
    memory.save_context({"dob": dob_text}, {"events": events_text})

    st.subheader("Biography")
    st.write(bio_text)

    st.subheader("Date of Birth")
    st.write(dob_text)

    st.subheader("Major Events")
    st.write(events_text)

    st.subheader("Stored Memory (All Outputs)")
    for msg in memory.chat_memory.messages:
        st.write(msg.content)
