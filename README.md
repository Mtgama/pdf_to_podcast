# 🎙️ تبدیل PDF به پادکست فارسی با استفاده از Coqui TTS و GPT

پروژه‌ای برای استخراج متن از فایل PDF و تبدیل آن به یک گفت‌وگوی پادکستی جذاب به زبان فارسی، سپس تولید صدای طبیعی از آن با استفاده از مدل‌های **Coqui TTS** فارسی.
---
## ▶️ ویدیوی معرفی پروژه

[![Watch on YouTube](https://s21.uupload.ir/files/matgama/github/ChatGPT%20Image%20Aug%207,%202025,%2011_27_12%20PM.png)](https://youtu.be/ll7KybZZFjw)

---

## 📸 پیش‌نمایش رابط کاربری

![App Screenshot](https://s21.uupload.ir/files/matgama/github/Screenshot%20from%202025-08-07%2023-52-13.png)

---
## 📸 مراحل کلی پروژه

![App Screenshot](https://s21.uupload.ir/files/matgama/github/Untitled%20Diagram.jpg)



---

## 🚀 شروع سریع

### 1. کلون کردن مخزن
```bash
git clone https://github.com/karim23657/Persian-tts-coqui.git
cd Persian-tts-coqui
```

###  2. ساخت محیط مجازی و نصب وابستگی‌ها
```bash
python -m venv venv
source venv/bin/activate  # یا در ویندوز: venv\Scripts\activate
```
### نصب نیازمندی ها:

```python
pip install -r requirements.txt
```

- برای استفاده از مدل صوتی فارسی آموزش‌دیده، مراحل زیر را انجام دهید:

1. دانلود فایل‌های مدل:
فایل	لینک دانلود:
 [دانلود](https://github.com/karim23657/Persian-tts-coqui)

📚 نحوه استفاده
1. اجرای اپلیکیشن Streamlit:
```python
streamlit run app.py
```

2. امکانات پروژه:
آپلود فایل PDF

تبدیل متن PDF به گفت‌وگوی پادکستی طبیعی (مثلاً بین علی و سارا)

تولید صدای گفت‌وگو با استفاده از مدل TTS فارسی

پخش یا دانلود فایل صوتی خروجی

🛠️ تکنولوژی‌های استفاده‌شده:

Python 3.10+

Streamlit – رابط کاربری تحت وب

Coqui TTS – تبدیل متن به گفتار

LangChain + GPT API – بازنویسی دیالوگی

PyPDF2 – استخراج متن از PDF




# English:

# 🎙️ Convert PDF to Persian Podcast Using Coqui TTS and GPT

A project to extract text from a PDF file and transform it into an engaging Persian podcast conversation, then generate natural-sounding speech using Persian **Coqui TTS** models.

---

## 📸 User Interface Preview

![App Screenshot](https://s21.uupload.ir/files/matgama/github/Screenshot%20from%202025-08-07%2023-52-13.png)

---

## 📸 General Project Workflow

![App Screenshot](https://s21.uupload.ir/files/matgama/github/Untitled%20Diagram.jpg)

---

## ▶️ Project Introduction Video

[![Watch on YouTube](https://img.youtube.com/vi/j-tf7VyxzjY/0.jpg)](https://www.youtube.com/watch?v=j-tf7VyxzjY)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/karim23657/Persian-tts-coqui.git
cd Persian-tts-coqui
```
2. Create a Virtual Environment and Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # or on Windows: venv\Scripts\activate
```
Install Requirements:
```python
pip install -r requirements.txt
```
📚 How to Use

Run the Streamlit Application:
```python
streamlit run app.py
```
