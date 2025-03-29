# ReviewRadar: Product Review Analysis with Natural Language Processing  
### *NUS GAIP 2024*

## Overview

**ReviewRadar** is a machine learning system designed to enhance the quality and credibility of product reviews on e-commerce platforms. Developed as part of the **Global Academic Internship Programme (GAIP)** at the **National University of Singapore (NUS)** held in the Summer of 2024. The project combines **Natural Language Processing (NLP)** and **supervised learning techniques** to solve two critical problems in online commerce:

1. **Sentiment Analysis** – to understand customer sentiment in product reviews.  
2. **Fake Review Detection** – to identify and filter computer-generated reviews that mislead consumers.

My team consisted of five members, each leading a different segment of the project. I focused on building and evaluating the classical machine learning pipeline for **fake review detection using Support Vecotr Machines**.

---

## Project Goals

- Improve consumer trust by filtering computer-generated reviews
- Help businesses extract real customer sentiment
- Compare traditional and neural models for performance and interpretability
- Build solutions that are practical for integration in real-world systems

---

## Models and Methodology (Team-Level)

We implemented and compared the performance of three modeling approaches across both tasks:

| Task                   | Model 1             | Model 2        |
|------------------------|---------------------|----------------|
| **Sentiment Analysis** | Logistic Regression | LSTM (RNN)     |
| **Fake Review Detection** | Support Vector Machine (SVM) | LSTM (RNN)     |

**LSTM models** showed state-of-the-art performance due to their ability to capture contextual dependencies in text. However, our classical models provided competitive accuracy with lower computational cost.

## Full Architecture

### Sentiment Analysis

- **Models**: Logistic Regression, LSTM
- **Embeddings**: Bag-of-Words (for LR), GloVe/Word2Vec (for LSTM)
- **Dataset**: Kaggle Amazon Review dataset (~53K reviews)
- **Accuracy**: Logistic Regression (~90%), LSTM (~89%)

### Fake Review Detection

- **Models**: Support Vector Machine (SVM), LSTM
- **Dataset**: 40K reviews (Salminen et al.)
- **Accuracy**: SVM (~87%), LSTM (~96%)


---

## Fake Review Detection with SVM

I led the entire pipeline for fake review classification using **Support Vector Machines** with **Bag-of-Words** features. This included:

### Dataset

- **40,000 Amazon product reviews**, with equal balance of real and fake entries.
- Fake reviews were generated using **GPT-2**, trained by Joni Salminen et al. ([dataset link](https://osf.io/tyue9/)).

### Data Preprocessing

Implemented a structured pipeline including:

- Lowercasing  
- Removal of special characters and digits  
- Tokenization  
- Stopword removal  
- Lemmatization

These steps ensured cleaner, normalised inputs for model training and improved generalization.

### Feature Engineering

Used **CountVectorizer** to implement the **Bag-of-Words model**:

- This involved creating a vocabulary of known words and then representing each review as a vector (of word frequencies) of its constituent words.

### Modeling with SVM

- Trained an `SVC` with a linear kernel.
- Applied **regularization (C parameter)** to balance margin maximization and training performance.

### Evaluation & Validation

- Performed **5-fold cross-validation** to improve generalization.
- Evaluated using **accuracy**, **precision**, **recall**, and **F1-score**.

| Metric         | Value |
|----------------|-------|
| Accuracy       | ~87%  |
| Precision (Fake) | ~87% |
| Recall (Fake)    | ~88% |
| F1-score (Fake)  | ~87% |

### Result Summary

The classical pipeline achieved high performance, validating that **SVM + BoW** serves as a strong baseline for fake review detection—even against more complex neural network approaches.

Additionally, I also contributed to the sentiment analysis task using logistic regression


### Areas for Improvement - SVM Fake Review Detection

- The dataset was sourced from a 2018 research study. Since then, large language models (LLMs) have advanced significantly, making AI-generated reviews more realistic and harder to distinguish from human-written content.

- This project employed a purely machine learning-based approach, and while the SVM classifier performed reasonably well, its decision-making process lacks interpretability. Combining this with rule-based heuristics or explainable AI (XAI) methods could enhance both accuracy and trust in predictions. Additionally, an integration of ML with other non-ML approaches (like metadata and behavioral pattern analysis, linguistic and stylometric rules, rule-based keyword filters, IP address and geolocation filtering, temporal and network analysis, verification and provenance checks etc.) is likely to lead to to a more robust and effective fake review detection system.

- The model was trained on a relatively limited and static dataset. With access to larger, more diverse datasets and greater computational resources, the model’s generalizability and robustness could be significantly improved.


