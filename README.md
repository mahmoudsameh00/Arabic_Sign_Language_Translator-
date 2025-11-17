# 🗣️ Arabic Sign Language Translator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Enabled-green)
![Gemini AI](https://img.shields.io/badge/Google%20Gemini-Powered-purple)


Description:

This project bridges the communication gap for the Arabic-speaking Deaf community by translating Sign Language into fluent text in real-time. Unlike standard translators that map gestures to static words, this system understands context.

It utilizes a Stacked LSTM neural network to recognize 100+ dynamic signs from MediaPipe hand landmarks. Unique to this project, it incorporates a Gender Detection module (OpenCV/Caffe) to ensure correct Arabic verb conjugation (e.g., distinguishing between "أنا ذاهبه" and "أنا ذاهب"). Finally, the disjointed words are processed by Google Gemini AI to generate grammatically perfect Arabic sentences, handling complex sentence structures that simple dictionary lookups cannot.

## 🌟 Key Features

* **Real-Time Detection:** Uses **MediaPipe** to track 84 hand keypoints at 15 FPS.
* **Deep Learning:** A custom **Stacked LSTM** model trained on 100+ Arabic Sign Language words.
* **Gender Awareness:** Integrated **OpenCV Face Detection** determines if the signer is Male or Female to apply correct Arabic verb conjugation (e.g., "يأكل" vs "تأكل").
* **AI Grammar Correction:** Utilizes **Google Gemini 2.0 Flash-lite** to convert disjointed words (e.g., "I school go") into grammatically perfect Arabic sentences (e.g., "أنا أذهب إلى المدرسة").
* **Optimistic UI:** Displays predicted words instantly while the LLM processes grammar in the background for a seamless user experience.

## 📂 Project Structure

Arabic-Sign-Language-Translator/
│
├── assets/                  # Images for Readme (screenshots, architecture diagrams)
├── data/                    # Raw npy data 
├── models/                  # Saved models (.h5, .caffemodel, .prototxt)
│   ├── action3.h5
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
│
├── notebooks/               # original experimental notebooks here
│   ├── Model_Words.ipynb
│   ├── Word_real_time_trial.ipynb
│   └── Word_real_time.ipynb
│
├── src/                     # Source code package
│   ├── __init__.py
│   ├── config.py            # Configuration (mappings, paths, constants)
│   ├── keypoints.py         # MediaPipe and extraction logic
│   ├── model_def.py         # Neural Network Architecture definition
│   ├── llm_grammar.py       # Gemini API logic
│   └── utils.py             # Visualization and Arabic text handling
│
├── train.py                 # Script to launch training
├── app.py                   # Script to launch real-time inference
├── requirements.txt         # Python dependencies
├── .gitignore               # Files to exclude from Git
├── .env                     # (GitIgnore this) Store API Keys here
└── README.md                # Project Documentation
