import time

import pymongo

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import psycopg2

s  = SentimentIntensityAnalyzer()

# Establish a connection to the MongoDB server
client = pymongo.MongoClient(host="mongodb", port=27017)

time.sleep(10)  # seconds

# Select the database you want to use withing the MongoDB server
db = client.twitter



from sqlalchemy import create_engine

pg = create_engine('postgresql://postgres:titanic22@postgresdb:5432/twitter', echo=True)

pg.execute('''
    CREATE TABLE IF NOT EXISTS tweets (
    text VARCHAR(500),
    sentiment NUMERIC
);
''')

docs = db.tweets.find()
for doc in docs:
    print(doc)
    text = doc['text']
    query = "INSERT INTO tweets VALUES (%s, %s);"
    sentiment = s.polarity_scores(doc['text'])  # assuming your JSON docs have a text field
    print(sentiment)
    # replace placeholder from the ETL chapter
    score = sentiment['compound']
    pg.execute(query, (text, score))
