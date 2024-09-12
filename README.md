# Suicidal Post Detection Using Machine Learning (RNN-Based)

## Motivation
The goal of this project is to develop a machine learning model that can predict whether a social media post indicates suicidal behavior, enabling early intervention and assistance for the author.

## Data Collection
The dataset was collected from three Kaggle sources, containing posts from Reddit and Twitter. Approximately 14% of the 101,357 posts are suicidal, while the remaining 86% are non-suicidal.

## Data Processing
The data preprocessing steps included:
- Removal of null rows and stopwords.
- Lemmatization to convert words to their base form.
- Conversion to lowercase to maintain uniformity.

## Feature Extraction
- **Word2Vec** embeddings were used to convert each word into a 300-dimensional vector. Posts were padded to ensure a uniform length of 100 words.
- **TF-IDF** was applied to avoid negative values, making the data suitable for the **Multinomial Naive Bayes** model.

## Model Creation
The data was split into an 80:20 ratio for training and testing. Three models were trained:
1. **Random Forest** with 200 estimators.
2. **Multinomial Naive Bayes** using TF-IDF vectorized data.
3. **GRU RNN** model:
   - First, the data was undersampled (20,968 training samples, 5,242 testing samples).
   - In the second run, a dataset was created where non-suicidal posts were twice the number of suicidal posts. The model used 3 GRU layers, 3 Dropout layers, and 1 Dense layer. Class weights were applied to handle imbalance.

## Performance
The following table summarizes the model performance, where **class 0** represents **non-suicidal posts** and **class 1** represents **suicidal posts**:

| Model                   | Accuracy | Class | Precision | Recall | F1 Score |
|--------------------------|----------|-------|-----------|--------|----------|
| Random Forest             | 0.95     | 0     | 0.96      | 0.98   | 0.97     |
|                          |          | 1     | 0.86      | 0.69   | 0.77     |
| Multinomial Naive Bayes   | 0.93     | 0     | 0.93      | 0.99   | 0.96     |
|                          |          | 1     | 0.91      | 0.48   | 0.63     |
| GRU (1 Layer)             | 0.91     | 0     | 0.90      | 0.93   | 0.91     |
|                          |          | 1     | 0.93      | 0.89   | 0.91     |
| GRU (Multiple Layers)     | 0.92     | 0     | 0.96      | 0.91   | 0.94     |
|                          |          | 1     | 0.84      | 0.93   | 0.88     |

## Challenges
Due to limited memory resources, it was not possible to feed the entire dataset into computationally heavy models like GRU.

## Future Work
1. Use a larger dataset to train the model.
2. Integrate the model into social media platforms.
3. If a post is predicted to be suicidal, the system will reach out to the author to offer support. If the author does not respond, it will attempt to contact the author's emergency contact.

