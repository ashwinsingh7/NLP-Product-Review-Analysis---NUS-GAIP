# ReviewRadar: Product Review Analysis with Natural Language Processing  
### *NUS GAIP 2024*

## Overview

This project, developed as part of the **Global Academic Internship Programme (GAIP)** at the **National University of Singapore (NUS)**, combines **Natural Language Processing (NLP)** and **supervised learning techniques** to address two critical problems in online commerce:

1. **Sentiment Analysis** – to understand customer sentiment in product reviews.  
2. **Fake Review Detection** – to identify and filter computer-generated reviews that mislead consumers.

My team consisted of five members, each leading a different segment of the project. I focused on building and evaluating the classical machine learning pipeline for fake review detection using Support Vector Machines, and also did some work on sentiment analysis using Logistic Regression.

Fake Review Detection colab notebook - https://drive.google.com/file/d/1wI_IW5JY2HD862A3Qapre1CMZbXxIYzt/view?usp=sharing

<img width="350" alt="Screenshot 2025-03-29 at 8 11 40 PM" src="https://github.com/user-attachments/assets/df870301-5359-444c-aad1-270e667d4a30" />


---

## Project Goals

- Improve consumer trust by filtering computer-generated reviews
- Help businesses extract real customer sentiment
- Compare classical ML models with neural network models for performance and interpretability
- Build solutions that are practical for integration in real-world systems

<img width="350" alt="Screenshot 2025-03-29 at 8 12 58 PM" src="https://github.com/user-attachments/assets/87fd7659-d0b5-4230-b886-d12c0bd154fc" />


---

## Models and Methodology (Team-Level)

We implemented and compared the performance of three modeling approaches across both tasks:

| Task                   | Model 1             | Model 2        |
|------------------------|---------------------|----------------|
| **Sentiment Analysis** | Logistic Regression | LSTM (RNN)     |
| **Fake Review Detection** | Support Vector Machine (SVM) | LSTM (RNN)     |

**LSTM models** showed state-of-the-art performance due to their ability to capture contextual dependencies in text. However, our classical models provided competitive accuracy with lower computational cost.

<img width="400" alt="Screenshot 2025-03-29 at 8 12 17 PM" src="https://github.com/user-attachments/assets/0a7e24cf-a998-41b7-91ee-9b1918a88be5" />


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

- **~40K Amazon product reviews**, with equal balance of real and fake entries.
- Fake reviews were generated using **GPT-2**, trained by Joni Salminen et al. ([dataset link](https://osf.io/tyue9/)) in their 
[2022 paper](https://www.sciencedirect.com/science/article/pii/S0969698921003374?via%3Dihub).

### Data Preprocessing

Implemented a structured pipeline including:

- Lowercasing  
- Removal of special characters and digits  
- Tokenization  
- Stopword removal  
- Lemmatization

These steps ensured cleaner, normalised inputs for model training and improved generalization.

<img width="400" alt="Screenshot 2025-03-29 at 8 12 02 PM" src="https://github.com/user-attachments/assets/979f2fad-831a-4b37-896d-d1a9afa01d84" />


### Feature Engineering

Used **CountVectorizer** to implement the **Bag-of-Words model**:

- This involved creating a vocabulary of known words and then representing each review as a sparse vector (comprising word frequencies of its constituent words).

### Modelling with SVM

- Trained an `SVC` with a linear kernel.
- Applied **regularization (C parameter)** to balance margin maximisation and training loss.

### Evaluation & Validation

- Performed **5-fold cross-validation** to improve generalization.
- Evaluated using **accuracy**, **precision**, **recall**, and **F1-score**.

| Metric         | Value |
|----------------|-------|
| Accuracy       | ~87%  |
| Precision (Fake) | ~87% |
| Recall (Fake)    | ~88% |
| F1-score (Fake)  | ~87% |

<img width="400" alt="Screenshot 2025-03-29 at 8 12 40 PM" src="https://github.com/user-attachments/assets/03c9e58f-c7a9-4eaa-a232-33a10c796bc4" />


### Result Summary

The pipeline achieved reasonably high performance, validating that **SVM + BoW** serves as a strong baseline for fake review detection.

Additionally, I also contributed to the sentiment analysis task using logistic regression and Bag-of-Words feature engineering.

<br>

### Future Exploration

Implement the [2022 paper](https://www.sciencedirect.com/science/article/pii/S0969698921003374?via%3Dihub) from which the dataset was taken and try to replicate their results and understand the theoretical underpinnings of the models they used.



### Areas for Improvement - SVM Fake Review Detection

- The dataset was sourced from a 2022 research study. Since then, large language models (LLMs) have advanced significantly, making AI-generated reviews more realistic and harder to distinguish from human-written content.

- This project employed a purely machine learning-based approach, and while the SVM classifier performed reasonably well, its decision-making process lacks interpretability. Combining this with rule-based heuristics or explainable AI (XAI) methods could enhance both accuracy and trust in predictions. Additionally, an integration of ML with other non-ML approaches (like metadata and behavioral pattern analysis, linguistic and stylometric rules, rule-based keyword filters, IP address and geolocation filtering, temporal and network analysis, verification and provenance checks etc.) is likely to lead to to a more robust and effective fake review detection system. One can also look into other models that may perform better than SVMs on the classification task.

- The model was trained on a relatively limited and static dataset. With access to larger, more diverse datasets and greater computational resources, the model’s generalizability and robustness could be significantly improved.


