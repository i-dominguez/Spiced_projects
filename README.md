![Image](https://github.com/i-dominguez/spiced_projects/blob/main/spiced_logo.png)

This repository contains the projects I completed during a Data Science Bootcamp at SPICED Academy in Berlin from June to September 2022.

## 01. Visual Data Analysis - Animated Scatterplot

![Image](https://github.com/i-dominguez/spiced_projects/blob/main/01_data_visualization/animated_plot.gif)

This animated scatterplot visualizes the changes of countries' fertility rate, life expactancy and population between 1850 and 2015. The size of the scatters represents the population of each country, the colours shows in which continent they can be found.

Data source: [Gapminder Foundation](https://www.gapminder.org/data/).

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/01_data_visualization).


## 02. Supervised Machine Learning: Classification - Kaggle's Titanic Challenge

The goal of this project was to built a machine learning model to predict the survival of Titanic passenger based on the features in the dataset of Kaggle's Titanic - Machine Learning from Disaster.

Based on the Exploratory Data Analysis (plotted missing values and the correlation between survival and the different data categories) selected the most significant features and dropped the ones which cannot contribute to accurate prediction.

In feature engineering using ColumnTransformer, I applied 1) OneHotEncoder: to convert categorical variables into binary features, 2) SimpleImputer: to fill missing values and 3) MinMaxScaler: to normalize continous numerical variable in range 0.0 - 1.0.

The data was trained on Scikit-learn's LogisticRegression and RandomForestClassifier models. After evaluating different model's accuracy scores and cross validation, I kept the RandomForest model for prediction (cross validation: mean accuracy score 82.06%).

Data source: [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview).

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/02_titanic_classification).

## 03. Supervised Machine Learning: Regression - Bicycle Rental Forecast

The goal of this project is to build a regression model, in order to predict the total number of rented bycicles in each hour based on time and weather features, optimizing the accuracy of the model for RMSLE, using Kaggle's "Bike Sharing Demand" dataset that provides hourly rental data spanning two years.

After exploratory data analysis (EDA) and extracting datetime features, highly correlated variables were dropped via feature selection (correlation analysis) to avoid multicollinearity. I then used the Linear Regression model and obtained predictions with 0.7289 RMSLE.

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/03_regression_bicycle_rental).

## 04. Natural Language Processing (NLP): Text Classification

The main goal of this project was to build a text classification model on song lyrics to predict the artist from a piece of text.

Through web scraping with BeautifulSoup, the song-lyrics of selected artists are extracted from lyrics.com. During the text pre-processing, word-tokenizer and word-lemmatizer of Natural Language Toolkit (NLTK) is used to "clean" the extracted texts and create the corpus: 1) TreebankWordTokenizer() splits the text into list of words and removes all other punctuation marks, 2) WordNetLemmatizer() reverts words back to their root/base. These steps are required to import and download lexical database such as WordNet. WordNet's Stopwords also removes the most common English words from the corpus.

In the model pipeline, Tfidfvectorizer (TF-IDF) transforms the words of the corpus into a matrix, count-vectorizes and normalizes them at once by default. For classification, the multinomial Naive Bayes classifier MultinomialNB() was used which is suitable for classification with discrete features like word counts for text classification.

Versions
I built 2 versions: - Version 1 extracts song lyrics directly from the htmls. - Version 2 extracts, downloads and saves song lyrics locally in separate .txt files. Then all lyrics will be loaded from the .txt files to create the corpus.

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/04_nlp_text_classification).

## 06. The Data Pipeline: Tweets Sentiment Analysis

![Image](https://github.com/i-dominguez/spiced_projects/blob/main/06_docker_etl_data_pipeline/twitter_slackbot/structure.svg)

The challenge of this Data Engineering project was to build a Dockerized Data Pipeline to analyze the sentiment of tweets. At first, using Tweepy API, tweets are collected in a selected topic and stored in a MondoDB database (tweet_collector). Next, the sentiment of tweets is analyzed and the tweets with the scores are stored in a Postgres database (ETL_job). Finally, tweets with sentiment score are published on a Slack channel.

For the sentiment analysis, SentimentIntensityAnalyzer() of the the Vader library (Valence Aware Dictionary and sEntiment Reasoner) was used.

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/06_docker_etl_data_pipeline/twitter_slackbot).

## 07. Time Series Analysis: Temperature Forecast

In this project, I applied the ARIMA model for a short-term temperature forecast. After visualizing the trend, the seasonality and the remainder of the time series data (daily mean temperature in Berlin-Tempelhof from 1979-2020), I run tests such as ADF for checking stationarity (time dependence).

Data source: [European Climate Assessment Dataset](https://www.ecad.eu/).

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/07_time_series).

## 08. Markov Chain Monte Carlo (MCMC): Predicting and simulating customer behaviour in a supermarket

The goal of this project was to predict and visualize customer behaviour between departments/aisles in a supermarket, applying Markov Chain modeling and Monte-Carlo simulation.

The project included the following tasks:

1. Data Analysis
2. Calculating Transition Probabilities between the aisles
3. Implementing a Customer Class
4. Running MCMC (Markov-Chain Monte-Carlo) simulation for a single class customer
5. Extending the simulation to multiple customers
6. Animation/Visualization

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/08_markov_chain_montecarlo).


## 09. Deep Learning - Artificial Neural Network

The goal of this project was to build an Artificial Neural Network that recognizes objects on images made by the webcam:

The project included the following tasks:

1. Implementing a Feed-Forward Neural Network
2. Backpropagation from Scratch
3. Building Neural Network with Keras
4. Training Strategies / Hyperparameters of Neural Networks
5. Convolutional Neural Networks (CNN)
6. Classifying images made with webcam with Pre-trained Networks (Comparing MobileNetV2, InceptionV3)
7. Image Detection

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/09_deep_learning).


## 10. Recommender systems - Movie Recommender with Collaborative Filtering

The movie recommender is based on the Collaborative Filtering approach, and creates predictions for movie ratings with Matrix Factorization technique (NMF). It is trained on the 'small' dataset of [MovieLens](https://grouplens.org/datasets/movielens/).

The online user-interface was built and deployed with Streamlit.

The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/10_recommender_system).


## Final project - Text difficulty classifier
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


The folder of this project can be found [here](https://github.com/i-dominguez/spiced_projects/tree/main/final_project).





