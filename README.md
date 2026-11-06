📈 Financial News Sentiment Prediction using Deep Learning & FinBERT
📌 Project Overview

Financial sentiment plays a critical role in investment decisions, market analysis, and risk management. This project aims to automatically classify finance-related tweets into Bearish, Bullish, or Neutral sentiments using both traditional deep learning models and a domain-specific transformer model (FinBERT).

The project follows a progressive approach:

Build baseline deep learning models (RNN, LSTM, GRU)
Compare their performance
Fine-tune FinBERT on the same dataset
Develop an interactive Streamlit web application for real-time sentiment prediction

The results demonstrate how transformer-based models significantly outperform traditional recurrent neural networks in financial text classification tasks.

🎯 Business Use Cases
Monitor investor sentiment in real time
Track bullish and bearish market trends
Support financial research teams with automated sentiment analysis
Build market mood indicators for trading dashboards
Analyze financial news headlines and social media discussions
🗂 Dataset Information

Dataset: Financial News Sentiment Dataset

Task: Multi-class Sentiment Classification

Sentiment Classes
Label	Sentiment
0	Bearish 📉
1	Bullish 📈
2	Neutral ➖
Dataset Split
Dataset	Records
Training	9,543
Validation	2,388
🧹 Data Preprocessing

The following preprocessing techniques were used:

For RNN/LSTM/GRU Models
Lowercasing text
URL removal
Email removal
Mention removal
Hashtag removal
Removal of special characters
Tokenization using NLTK
Lemmatization using WordNetLemmatizer
Vocabulary creation
Sequence encoding
Padding
For FinBERT

Minimal preprocessing was applied:

URL removal
Mention removal
Whitespace normalization

Financial tickers and domain-specific terminology were preserved to maintain contextual meaning.

🤖 Models Implemented
1. Simple RNN

A basic recurrent neural network was implemented as the first baseline model.

Architecture

Plain Text
Embedding Layer
↓
Simple RNN
↓
Dense Layer
↓
Output Layer
2. LSTM

Long Short-Term Memory networks were used to address the vanishing gradient problem present in standard RNNs.

Architecture

Plain Text
Embedding Layer
↓
LSTM
↓
Dropout
↓
Dense Layer
↓
Output Layer
3. GRU

Gated Recurrent Units provide similar capabilities to LSTM with fewer parameters.

Architecture

Plain Text
Embedding Layer
↓
GRU
↓
Dropout
↓
Dense Layer
↓
Output Layer
4. FinBERT

FinBERT is a finance-specific version of BERT pretrained on financial text and further fine-tuned on the sentiment dataset.

Architecture

Plain Text
Financial Text
↓
FinBERT Tokenizer
↓
FinBERT Transformer
↓
Classification Head
↓
Sentiment Prediction
📊 Model Performance Comparison
Model	Accuracy	Macro F1 Score
RNN	59.55%	38.32%
GRU	78.56%	68.80%
LSTM	79.27%	70.73%
FinBERT	87.23%	82.45%
🏆 Best Model
FinBERT

Validation Metrics

Metric	Score
Accuracy	87.23%
Precision	83.23%
Recall	81.75%
Macro F1	82.45%
Why FinBERT Performed Better

FinBERT is pretrained on financial language and understands domain-specific terminology such as:

earnings
guidance
downgrade
upgrades
bullish
bearish
revenue miss
market outlook

This allows FinBERT to capture contextual meaning more effectively than models trained from scratch.

🌐 Streamlit Application

An interactive Streamlit dashboard was developed for real-time financial sentiment prediction.

Features

✅ Modern FinTech-style UI

✅ Real-time sentiment prediction

✅ Probability visualization

✅ Sample financial headlines

✅ Model comparison dashboard

✅ FinBERT-powered inference

User Workflow
Plain Text
Financial Headline
↓
FinBERT Model
↓
Prediction
↓
Sentiment Class
↓
Confidence Scores
🛠 Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Seaborn
NLTK
PyTorch
Hugging Face Transformers
Datasets
Scikit-Learn
Streamlit
Plotly
📂 Project Structure
Plain Text
Financial_Sentiment_Analysis/
 
│
├── data/
│ ├── sent_train.csv
│ └── sent_valid.csv
│
├── models/
│ ├── best_lstm_model.pt
│ └── vocab.pkl
│
├── notebooks/
│ ├── Financial_Sentiment.ipynb
│ ├── FinBERT.ipynb
│ └── finbert_model/
│
├── app/
│ └── app.py
│
├── reports/
│
├── requirements.txt
│
└── README.md
🚀 How to Run the Project
Clone Repository
Shell
git clone <repository_url>
Install Dependencies
Shell
pip install -r requirements.txt
Launch Streamlit App
Shell
streamlit run app.py
🔮 Future Improvements
Deploy using Streamlit Cloud
Introduce model explainability (SHAP/LIME)
Add support for financial news articles
Integrate live Twitter/X API feeds
Build a market sentiment dashboard
Deploy as a cloud-hosted REST API
✅ Conclusion

This project successfully demonstrates the effectiveness of deep learning and transformer-based architectures for financial sentiment analysis. Among all implemented models, FinBERT achieved the highest performance with 87.23% validation accuracy and 82.45% macro F1 score, outperforming traditional RNN-based approaches by a significant margin.

The final solution includes an end-to-end machine learning pipeline, model comparison study, and a production-style Streamlit application capable of performing real-time financial sentiment prediction.

👨‍💻 Author

Binduja

Financial News Sentiment Prediction using Deep Learning & FinBERT 🚀