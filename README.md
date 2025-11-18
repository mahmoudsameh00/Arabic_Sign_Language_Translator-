# 🗣️ Arabic Sign Language Translator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Enabled-green)
![Gemini AI](https://img.shields.io/badge/Google%20Gemini-Powered-purple)

This project bridges the communication gap for the Arabic-speaking Deaf community by translating Sign Language into fluent text in real-time. Unlike standard translators that map gestures to static words, this system understands context.

It utilizes a Stacked LSTM neural network to recognize 100+ dynamic signs from MediaPipe hand landmarks. Unique to this project, it incorporates a Gender Detection module (OpenCV/Caffe) to ensure correct Arabic verb conjugation (e.g., distinguishing between "أنا ذاهبه" and "أنا ذاهب"). Finally, the disjointed words are processed by Google Gemini AI to generate grammatically perfect Arabic sentences, handling complex sentence structures that simple dictionary lookups cannot.

## 🌟 Key Features

* **Data Collection Pipeline:** Includes a built-in tool to record, label, and augment sign language datasets using your own webcam.
* * **Deep Learning:** A custom **Stacked LSTM** model trained on 100+ Arabic Sign Language words.
* **Real-Time Detection:** Uses **MediaPipe** to track 84 hand keypoints at 15 FPS.
* **Gender Awareness:** Integrated **OpenCV Face Detection** determines if the signer is Male or Female to apply correct Arabic verb conjugation (e.g., "يأكل" vs "تأكل").
* **AI Grammar Correction:** Utilizes **Google Gemini 2.0 Flash-lite** to convert disjointed words (e.g., "I school go") into grammatically perfect Arabic sentences (e.g., "أنا أذهب إلى المدرسة").

## 🛠️ Architecture

1.  **Data Collection Pipeline:**
    * **Capture:** The system includes a dedicated tool (`collecting_data.ipynb`) that accesses the webcam to record custom datasets. It captures **30 videos** per sign, with each video containing **30 frames** of motion.
    * **Feature Extraction:** For every frame, **MediaPipe** extracts 84 distinct keypoints (x, y coordinates) from both hands, converting raw pixels into structured numerical data.
    * **Optimization:** To reduce computational cost and latency, the system subsamples the input to **15 frames** per sequence before training. This effectively captures the full temporal dynamics of the gesture while halving the processing load.

2.  **Building the Model (Deep Learning):**
    * **Architecture:** The core is a custom **Stacked LSTM (Long Short-Term Memory)** neural network, specifically chosen for its ability to learn patterns in time-series data (motion over time).
    * **Layer Structure:**
        * **Input Layer:** Accepts the sequence shape of `(15 frames, 84 keypoints)`.
        * **LSTM Layers:** Three progressive LSTM layers (64, 128, 64 units) process the temporal dependencies of the gestures.
        * **Regularization:** Dropout layers (0.4, 0.2) and L2 kernel regularization are embedded to prevent overfitting on the training data.
        * **Classification:** Fully connected Dense layers reduce the data dimensionality, ending in a Softmax layer that outputs probabilities across 100+ different sign classes.

3.  **Real-Time Detection:**
    * **Inference Loop:** The application runs at ~15 FPS, continuously filling a sliding buffer. It processes the last 15 frames of keypoints to make a prediction.
    * **Prediction & Gating:** The LSTM model predicts the current sign based on this buffer. A prediction is only accepted as valid if the confidence score exceeds **95%** (Threshold), effectively filtering out noise and idle movements.
    * **Asynchronous Grammar Correction:**
        * **Optimistic UI:** The recognized raw word (e.g., "eat") appears on screen immediately for instant feedback.
        * **Contextual Refinement:** A background thread simultaneously sends the accumulated words and the user's gender (detected via OpenCV) to **Google Gemini**. The LLM returns a grammatically correct Arabic sentence (e.g., converting "eat" to "I am eating" based on gender contexts), which seamlessly updates the display without causing video lag.




## 📂 Project Structure

```text
Arabic-Sign-Language-Translator/
│
├── assets/                  # Images for Readme
├── data/                    # Processed .npy sequence data
├── tirbo_data_images/       # (Optional) Raw images from data collection
│
├── models/                  # Saved models
│   ├── action3.h5           # Trained LSTM Model
│   ├── deploy.prototxt      # Face detection config
│   ├── res10_300x300...     # Face detection weights
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
│
├── notebooks/               # Notebooks for experiments & tools
│   ├── collecting_data.ipynb # Tool to record new signs
│   ├── Model_Words.ipynb    # Training experiments
│   └── Word_real_time.ipynb # Inference experiments
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
├── Arial.ttf                # Required font for rendering Arabic text
├── requirements.txt         # Python dependencies
├── .env                     # API Keys
└── README.md                # Project Documentation


```
## 🚀 Installation

1.  **Clone the repo:**
```bash
git clone [https://github.com/mahmoudsameh00/Arabic_Sign_Language_Translator-.git](https://github.com/mahmoudsameh00/Arabic_Sign_Language_Translator-.git)
cd sign-language-translator
 ```

2.  **Install dependencies:**
 ```bash
 pip install -r requirements.txt
 ```

3.  **Setup API Key:**
 * Create a .env file and add your Google Gemini API key:
     ```Ini, TOML
     GOOGLE_API_KEY=your_gemini_api_key_here
     ```

4.  **Run the Translator:**
 ```bash
 python app.py
 ```
* q: Quit the application.
* c: Clear the current sentence.

---

## 🧠 Training

If you wish to expand the vocabulary or retrain the model on your own dataset, follow this pipeline:

### 1. Collect Data
Use the included tool to record new signs via your webcam.
1.  Open `notebooks/collecting_data.ipynb`.
2.  Update the `actions` list with the new word(s) (e.g., `actions = np.array(['Hello'])`).
3.  Run the script. It will record **30 sequences** per action and automatically generate **flipped augmentations**, effectively doubling your dataset size.
   
**We trained our model on a large dataset of
video recordings featuring people gesturing 100
various words which consist of 300 videos in each
word and each video is labeled with its
corresponding word.**
### 2. Configure Labels
If you added new signs, you must update the configuration file to map them correctly.
* Open `src/config.py`.
* Add your new action strings to the `ACTIONS` list.
* *(Optional)* Add the Arabic translation to `ARABIC_MAPPING` for the UI display.

### 3. Train the Model
Execute the training script to compile the LSTM network:
```bash
python train.py

```
## 🤝 Acknowledgements
* [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker): For efficient on-device hand tracking.
* [Google Gemini](https://ai.google.dev/): For advanced linguistic processing.
* [OpenCV](https://opencv.org/): For computer vision tasks.
