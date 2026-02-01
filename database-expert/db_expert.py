# pip install streamlit langchain langchain-openai langchain-core

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64
from dotenv import load_dotenv

load_dotenv()

st.title("Database Schema Analyzer 🗃️")
st.write("Upload your database schema diagram and ask questions about it!")

# File uploader
uploaded_file = st.file_uploader("Choose a schema image...", type=['png', 'jpg', 'jpeg', 'gif'])

if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption='Uploaded Schema', use_container_width=True)

    # Encode image
    base64_image = base64.b64encode(uploaded_file.read()).decode()

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Analyze button
    if st.button("Analyze Schema"):
        with st.spinner("Analyzing schema..."):
            messages = [
                SystemMessage(content="You are a database expert. Analyze this schema diagram."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Analyze this database schema comprehensively:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                )
            ]

            response = llm.invoke(messages)
            st.write("### Analysis:")
            st.write(response.content)

    # Question input
    st.write("---")
    question = st.text_input("Ask a question about the schema:")

    if question:
        with st.spinner("Thinking..."):
            messages = [
                SystemMessage(content="You are a database expert."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"Question: {question}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                )
            ]

            response = llm.invoke(messages)
            st.write("### Answer:")
            st.write(response.content)