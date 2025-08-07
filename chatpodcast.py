import streamlit as st
import PyPDF2
from TTS.config import load_config
from TTS.utils.synthesizer import Synthesizer
import torch
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import tempfile


llm = ChatOpenAI(
    model="gpt-4o-mini",  
    base_url="https://api.avalai.ir/v1",
    api_key="aa-vljXOzFhI2c4IdLtR6wUvQlcomoUSzbrPBFQoCACREXcor0A", 
)


@st.cache_resource
def load_tts_model():

    config_path = "/home/mtgama/Desktop/smartglass/TTS/config1.json"
    model_path = "/home/mtgama/Desktop/smartglass/TTS/best_model_30824.pth"
    synthesizer = Synthesizer(model_path, config_path)
    return synthesizer


synthesizer = load_tts_model()
synthesizer.tts("سلام بر شما ای دوست من")

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.strip()


def convert_to_podcast_dialogue(text):
    messages = [
        SystemMessage(
            content="متن ورودی از یک فایل PDF است. آن را به گفت‌وگوی پادکستی بین دو نفر بازنویسی کن (مثلاً علی و سارا) با لحن طبیعی.لطفا از هیچ علایم نگارشی ای استفاده نکن و فقط متن مجازه و کاراکتر نقطه = ."
        ),
        HumanMessage(content=text)
    ]
    return llm.invoke(messages).content


st.title("🎧 ساخت پادکست از PDF با GPT-4o-mini + TTS")

pdf_file = st.file_uploader("یک فایل PDF آپلود کن", type="pdf")

if pdf_file:
    with st.spinner("در حال استخراج متن..."):
        raw_text = extract_text_from_pdf(pdf_file)
        st.success("✅ متن استخراج شد!")

    st.text_area("📄 متن استخراج‌شده:", raw_text, height=200)

    if st.button("تبدیل به گفت‌وگوی پادکستی"):
        with st.spinner("در حال تولید دیالوگ با GPT..."):
            dialogue = convert_to_podcast_dialogue(raw_text)
            st.success("✅ دیالوگ تولید شد!")
            st.text_area("🗣 دیالوگ پادکستی:", dialogue, height=300)

            with st.spinner("در حال تبدیل به صدا..."):
                audio = synthesizer.tts(dialogue)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                    synthesizer.save_wav(audio, tmp_wav.name)
                    st.audio(tmp_wav.name, format="audio/wav")
                    st.success("🔊 فایل صوتی آماده است!")
