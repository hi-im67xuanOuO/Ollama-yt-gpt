from langchain.globals import set_debug
from langchain.globals import set_verbose

#StreamLit Dependencies
import streamlit as st
from streamlit_chat import message

#Environment Dependencies
from dotenv import load_dotenv
import os

from youtube_transcript_api import YouTubeTranscriptApi

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS




def find_insights(video_link):

    if os.path.exists('transcripts'):
        print('Directory already exists')
    else:
        os.mkdir('transcripts')
    
    video_id = video_link.split('=')[1]
    dir = os.path.join('transcripts',video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    with open(dir+'.txt', 'w') as f:
        for line in transcript:
            f.write(f"{line['text']}\n")

    transcripts = ""
    for tran in transcript:
        transcripts += tran['text'] + " "




    llm = Ollama(model='llama2')
    embeddings = OllamaEmbeddings()

    vector = FAISS.from_texts([transcripts], embeddings)
    retriever = vector.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ('system', 'Answer the user\'s questions based on the below context:\n\n{context}'),
        ('user', '{input}'),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    chat_history = []
    response = retrieval_chain.invoke({
        'input': "Give me an abstract of the transcription.",
        'chat_history': chat_history,
    })
    return response['answer']







import streamlit as st
import os
import re

st.set_page_config(page_title="📹 YT-GPT")


def youtube_video_url_is_valid(url: str) -> bool:
    pattern = r'^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&ab_channel=[\w\d]+)?$'
    match = re.match(pattern, url)
    return match is not None

def yt_gpt_app():
    st.header("📹 YT-GPT : Generate insights of YouTube video")

    st.write(
    """
    A Streamlit app for extracting insights from YouTube videos using LangChain and Ollama LLM. Enter video URLs to get quick summaries and key details without watching."""
    )
    st.info(
        """
        Welcome! 👋 The YT-GPT is a Streamlit-powered application designed to provide quick and meaningful insights from YouTube videos. By simply entering the URL of a YouTube video, users can extract valuable information from its transcript using advanced LLM techniques. ✨
        """,
        icon="👾",
    )
        
    youtube_video_url = st.text_input(" Youtube video URL : ",
                                      placeholder="https://www.youtube.com/watch?v=**********")
    

    if not youtube_video_url_is_valid(youtube_video_url):
        st.error("Please enter a valid youtube video URL.")
        
    if st.button("Get insights"):
        if not youtube_video_url:
            st.warning("Please enter the Youtube vide URL")
        else:
            with st.spinner("Getting insights about the youtube video..."):
                answer = find_insights(youtube_video_url)
            st.success(answer)


yt_gpt_app()
