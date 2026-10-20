import re
import validators
import streamlit as st
import requests
from bs4 import BeautifulSoup
import yt_dlp
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
Provide a concise summary of the following content in about 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def extract_youtube_content(url):
    """Extracts transcript text using yt-dlp, with a fallback to video title & description if blocked."""
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'hi', 'es', 'fr', 'de', 'it', 'pt'],
        'socket_timeout': 10,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', '')
            description = info.get('description', '')
            
            # Try to find subtitle tracks from extracted info
            subtitles = info.get('subtitles') or info.get('automatic_captions')
            transcript_text = ""
            
            if subtitles:
                # Pick the first available language track URL
                for lang in ['en', 'hi', 'es', 'fr', 'de', 'it', 'pt']:
                    if lang in subtitles:
                        formats = subtitles[lang]
                        # Look for json3 or vtt format
                        sub_url = None
                        for fmt in formats:
                            if fmt.get('ext') in ['json3', 'vtt', 'srv1']:
                                sub_url = fmt.get('url')
                                break
                        if sub_url:
                            sub_res = requests.get(sub_url)
                            if sub_res.status_code == 200:
                                # Simple cleanup of VTT/JSON subtitle text
                                raw_sub = sub_res.text
                                # Strip out timestamps/tags if VTT
                                lines = [line.strip() for line in raw_sub.split('\n') 
                                         if line.strip() and '-->' not in line and not line.strip().isdigit()]
                                transcript_text = " ".join(lines)
                                break
            
            # Fallback: If no subtitles could be downloaded, use Title + Description
            if not transcript_text.strip():
                transcript_text = f"Title: {title}\n\nDescription: {description}"
                
            return transcript_text
            
        except Exception as e:
            # Ultimate fallback to requests/BeautifulSoup on watch page metadata if yt-dlp is blocked
            raise RuntimeError(f"YouTube extraction failed: {e}")

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

                # 1. Handle YouTube URLs using yt-dlp
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    text = extract_youtube_content(generic_url)

                # 2. Handle Website URLs using requests & BeautifulSoup
                else:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    response = requests.get(generic_url, headers=headers)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = " ".join([p.get_text() for p in soup.find_all("p")])
                    
                    # Fallback if no <p> tags found
                    if not text.strip():
                        text = soup.get_text()

                # Split text into chunks to safely handle long transcripts/articles
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
                docs = text_splitter.create_documents([text])

                # Use map_reduce if content is large, stuff if it's short
                chain_type = "map_reduce" if len(docs) > 1 else "stuff"

                chain = load_summarize_chain(
                    llm, 
                    chain_type=chain_type, 
                    prompt=prompt
                )
                response_output = chain.invoke({"input_documents": docs})
                
                # Extract output text from response dictionary
                output_summary = response_output.get("output_text", str(response_output))

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
