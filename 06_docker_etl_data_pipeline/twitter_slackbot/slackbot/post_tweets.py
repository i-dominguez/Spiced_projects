import time
import requests
from sqlalchemy import create_engine
import os
import psycopg2
import logging

webhook_url = 'https://hooks.slack.com/services/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'
#webhook_url = os.getenv('WEBHOOK_URL')
print(webhook_url)

time.sleep(10)


pg = create_engine('postgresql://postgres:titanic22@postgresdb:5432/twitter', echo=True)


get_tweets_highest = """
    select * from tweets
    LIMIT 5;
    """
    
get_tweets_highest = pg.execute(
    get_tweets_highest)
    

for t in get_tweets_highest:
    t._asdict()
    data = {'text': t['text'] + "\n The sentiment score of the tweet is:" + " " + str(t['sentiment'])}
    print(data)
    logging.critical(data)
    requests.post(url=webhook_url, json = data)

    



