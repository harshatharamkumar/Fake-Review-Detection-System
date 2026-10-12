# 🕵️ Fake Review Detection System

An AI-powered web app that classifies product reviews as **authentic** or **fake**, built with an **LSTM neural network** (TensorFlow/Keras) and served through a clean **Streamlit** interface.

Paste any product review, click **Analyze**, and get a verdict, a confidence score, and a visual gauge in seconds.

---

## ✨ Features

- **LSTM-based text classification** trained to distinguish genuine from fake reviews
- **Identical preprocessing to training** (lowercasing, URL removal, punctuation removal, stopword removal, lemmatization) so predictions stay consistent
- **Confidence scoring** with an interactive Plotly gauge
- **Preprocessing transparency**: see the original vs. cleaned text side by side
- **Risk indicators**: confidence level (High / Medium / Low) and risk level
- **Graceful error handling** for missing model files, missing TensorFlow, or missing NLTK data
- Polished, responsive UI with custom CSS

---

## 🧠 How It Works

```
Raw review text
      │
      ▼
Preprocessing  ──►  lowercase → remove URLs → remove punctuation
                    → remove stopwords → lemmatize
      │
      ▼
Keras Tokenizer  ──►  text → integer sequences
      │
      ▼
pad_sequences    ──►  fixed length (max_length, post-padding)
      │
      ▼
LSTM model       ──►  probability output
      │
      ▼
Label encoder    ──►  0 = Authentic, 1 = Fake  (+ confidence score)
```

- **Binary output**: confidence is the model's probability for the predicted class (threshold 0.5).
- **Multi-class output** is also supported through `argmax`.
- **Label convention**: `1` = Fake review, `0` = Authentic review.

---

## 🗂️ Project Structure

```
fake-review-detector/
├── app.py                                # Streamlit application
├── requirements.txt                      # Python dependencies
├── best_lstm_model.h5                    # Trained LSTM model
├── best_lstm_model_tokenizer.pkl         # Fitted Keras tokenizer
├── best_lstm_model_label_encoder.pkl     # Fitted label encoder
├── best_lstm_model_params.pkl            # Training params (e.g. max_length)
└── README.md
```

> ⚠️ The four `best_lstm_model*` files must sit in the **same folder as `app.py`**. They are produced by your training notebook/script. If they exceed GitHub's 100 MB file limit, use [Git LFS](https://git-lfs.com/).

---

## 🛠️ Tech Stack

| Purpose | Library |
|---|---|
| Web UI | Streamlit |
| Deep learning | TensorFlow / Keras (LSTM) |
| NLP preprocessing | NLTK |
| Data handling | NumPy, Pandas |
| Visualization | Plotly |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9 – 3.11** (recommended for TensorFlow compatibility)
- **pip**
- **VS Code** with the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) (optional but recommended)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fake-review-detector.git
cd fake-review-detector
```

### 2. Create and activate a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
streamlit
numpy
pandas
plotly
nltk
tensorflow
scikit-learn
```

> `scikit-learn` is needed to unpickle the label encoder. Use the same scikit-learn and TensorFlow versions you trained with to avoid loading errors. For a lighter install without a GPU, use `tensorflow-cpu`.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 💻 Running in VS Code

1. Open the project folder: **File → Open Folder…**
2. Open the terminal: **Ctrl + `** (backtick).
3. Select the interpreter: press **Ctrl + Shift + P** → **Python: Select Interpreter** → choose the `venv` you created.
4. Make sure the terminal shows `(venv)`. If not, activate it as shown above.
5. Run:
   ```bash
   streamlit run app.py
   ```
6. Hold **Ctrl** and click the `http://localhost:8501` link, or open it in your browser.

> Do **not** run it with the ▶ "Run Python File" button. Streamlit apps must be launched with `streamlit run`.

---

## 📖 Usage

1. Launch the app and click **🚀 Load Detection Model**.
2. Paste a product review into the text box.
3. Click **🔍 Analyze Review**.
4. Review the results:
   - Original vs. preprocessed text
   - Verdict: **Authentic Review** or **Fake Review Detected**
   - Confidence percentage and gauge
   - Stats: character count, processed word count, confidence level, risk level

### Example

| Input | Expected behavior |
|---|---|
| *"This product is absolutely amazing! Best purchase ever made. 5 stars definitely recommend to everyone!!!"* | Often flagged as suspicious, depending on the model |

---

## 🧪 Model Training (Reference)

Train your own model using the same preprocessing pipeline as the app, then save the artifacts:

```python
model.save("best_lstm_model.h5")

import pickle
pickle.dump(tokenizer,     open("best_lstm_model_tokenizer.pkl", "wb"))
pickle.dump(label_encoder, open("best_lstm_model_label_encoder.pkl", "wb"))
pickle.dump({"max_length": max_length}, open("best_lstm_model_params.pkl", "wb"))
```

If you change the preprocessing steps in training, update the `TextPreprocessor` class in `app.py` to match.

| Metric | Value |
|---|---|
| Accuracy | _add your result_ |
| Precision | _add your result_ |
| Recall | _add your result_ |
| F1-score | _add your result_ |

---

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| `Missing files: best_lstm_model.h5 ...` | Place all four model files next to `app.py`. |
| `TensorFlow not available` | `pip install tensorflow` (or `tensorflow-cpu`). |
| `LookupError: punkt` / `punkt_tab` not found | Run `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"` |
| `streamlit` not recognized | Activate your virtual environment and reinstall requirements. |
| Model fails to load / version errors | Install the same TensorFlow and scikit-learn versions used in training. |
| "Text became empty after preprocessing" | The review contained only stopwords/punctuation. Try longer text. |

---

## ⚠️ Disclaimer

This tool provides a **probabilistic prediction**, not proof. Results depend on the training data and may produce false positives or false negatives. Use it as a decision aid, not as the sole basis for judging a review.

---

## 🔮 Future Improvements

- Batch analysis of CSV files of reviews
- Transformer-based models (BERT) for higher accuracy
- Explainability (highlight words that influenced the prediction)
- Docker support and cloud deployment (Streamlit Community Cloud / Hugging Face Spaces)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
