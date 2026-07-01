# LogiCheck — Logical Fallacy Detection System

A context-aware system built with **DeBERTa-v3** and **Gemini AI** to detect and explain logical fallacies in text. 

## Features
- **Multi-Class Detection**: Identifies 8 common logical fallacies (e.g., Appeal to Authority, False Dilemma) or valid reasoning using a fine-tuned DeBERTa model.
- **AI Explanations**: Generates clear, educational explanations for detected fallacies via Gemini API.
- **Interactive UI**: Streamlit dashboard for data exploration, model metrics, and real-time prediction.
- **Focal Loss**: Handles extreme class imbalance in training data.

## Project Structure
```
nlp-code/
├── streamlit_app.py      # Streamlit web application
├── config.yaml           # Model & training configuration
├── requirements.txt      # Python dependencies
├── data/                 # CoCoLoFa dataset splits (Train/Dev/Test)
├── src/                  # Core NLP package
│   ├── data.py           # Preprocessing
│   ├── model.py          # DeBERTa-v3 Classifier
│   ├── explainer.py      # Gemini AI integration
│   ├── pipeline.py       # Inference pipeline
│   ├── train.py          # Training script
│   └── training_engine.py# Focal loss & Trainer
└── notebooks/            # Exploratory data analysis
```

## Quickstart

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the Model** (Dataset: CoCoLoFa)
```bash
python src/train.py --config config.yaml --data-dir data/
```

**3. Run the Dashboard**
```bash
streamlit run streamlit_app.py
```

## Team
- Anandhio Varistama - 2802455874
- Ganesha Chandra Abiwardhana - 2802446913
- Ivan Novanto Bastian - 2802457923
- Jason Tirta - 2802450715
- Muhammad Rizki Akbar - 2802456681
