import re
import validators
import streamlit as st
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain

# Streamlit APP Config
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and url (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Define Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def extract_youtube_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

if st.button("Summarize the Content from YT or Website"):
    # Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video url or website url")
    else:
        try:
            with st.spinner("Loading and summarizing content..."):
                # Initialize ChatGroq inside the button handler
                llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)

                # 1. Handle YouTube URLs with multi-language fallback
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = extract_youtube_video_id(generic_url)
                    if not video_id:
                        raise ValueError("Could not extract YouTube Video ID from the URL.")
                    
                    ytt_api = YouTubeTranscriptApi()
                    try:
                        # Try preferred languages first
                        fetched_transcript = ytt_api.fetch(video_id, languages=['en', 'hi', 'es', 'fr', 'de', 'it', 'pt'])
                        transcript_data = fetched_transcript.to_raw_data()
                    except Exception:
                        # Fallback: grab any available transcript if preferred languages fail
                        transcript_list = ytt_api.list(video_id)
                        first_transcript = next(iter(transcript_list))
                        fetched_transcript = first_transcript.fetch()
                        transcript_data = fetched_transcript.to_raw_data()
                    
                    text = " ".join([item['text'] for item in transcript_data])
                    docs = [Document(page_content=text, metadata={"source": generic_url})]

                # 2. Handle Website URLs using requests & BeautifulSoup
                else:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    response = requests.get(generic_url, headers=headers)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = " ".join([p.get_text() for p in soup.find_all("p")])
                    
                    # Truncate to first 6,000 characters to keep it fast and within TPM limits
                    text = text[:6000]
                    docs = [Document(page_content=text, metadata={"source": generic_url})]

                # Fast single-call summarization chain
                chain = load_summarize_chain(
                    llm, 
                    chain_type="stuff", 
                    prompt=prompt
                )
                response = chain.invoke({"input_documents": docs})
                
                # Extract output text from response dictionary
                output_summary = response.get("output_text", str(response))

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")