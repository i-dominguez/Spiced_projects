## 06. The Data Pipeline: Tweets Sentiment Analysis

The challenge of this Data Engineering project was to build a Dockerized Data Pipeline to analyze the sentiment of tweets. At first, using Tweepy API, tweets are collected in a selected topic and stored in a MondoDB database (tweet_collector). Next, the sentiment of tweets is analyzed and the tweets with the scores are stored in a Postgres database (ETL_job). Finally, tweets with sentiment score are published on a Slack channel.

For the sentiment analysis, SentimentIntensityAnalyzer() of the the Vader library (Valence Aware Dictionary and sEntiment Reasoner) was used.
