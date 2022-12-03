In this final project, I built a text classifier that predicts the difficulty of a given English text according to the reference levels [A1-A2, B1-B2, C1-C2] from the Common European Framework of Reference for Languages (CEFR).

Using a [dataset](https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts) from Kaggle that contains about 1,500 labeled texts, I followed the following steps:

1. **Data cleaning**:
Removal of punctuation, numbers and line breaks. I also made the text lower case, with the help of regular expressions (regex).

2. **Preprocessing**:
Word-tokenizer and word-lemmatizer of Natural Language Toolkit (NLTK) were used to "clean" the extracted texts and create the corpus: 1) TreebankWordTokenizer() splits the text into list of words, 2) WordNetLemmatizer() reverts words back to their root/base. These steps are required to import and download lexical database such as WordNet. WordNet's Stopwords also removes the most common English words from the corpus.

3. **Vectorization**:
Tfidfvectorizer (TF-IDF) transforms the words of the corpus into a matrix, count-vectorizes and normalizes them at once by default. 

4. **Oversampling**:
The EDA showed that the dataset had a class imbalance, as the number of texts per class (label) is different. To deal with this, oversampling was used.

5. **Model testing**:
I trained different models, such as logistic regression, random forest, naive bayes and SVM. The model which resulted in the highest accuracy and precision score was multinomial logistic regression, with 84%.

6. **Deployment**:
I pickled the best model and then build and deployed it on [Streamlit](https://i-dominguez-text-difficulty-app-l9fuvk.streamlitapp.com/).
