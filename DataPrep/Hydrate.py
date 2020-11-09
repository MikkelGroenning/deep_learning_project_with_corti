
from twarc import Twarc
import pandas as pd
import tweepy
from tqdm import trange

df = pd.read_csv("../Data/Interim/Tweet200316.tsv")

api_key = "vhpYxpHap3JT2knXp3hyiUrJs"
api_secret_key = "TVsN2J4ZrHEHwrPlhtihuBWolXWibF7878exrK9hQ7zQywq9Dp"

bearer_token = "AAAAAAAAAAAAAAAAAAAAAGe7JQEAAAAAxY1bj4PHloQmCG6MXMGiKVoUlyM%3DwLgIydkepC59w4RJ54ghNEuznPUDQkC2yhwr1deQCwwQY02V5L"
access_token = "1324015519708684289-HBCPYHbS8HPZ0Rq4YY5ZarJvOmei8I"
access_token_secret = "6NO0tM6dMoapFyCLLFt2OaQJ1xmvO1sUoUmrPJdmmquXx"

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)

try:
    redirect_url = auth.get_authorization_url()
except tweepy.TweepError:
    print('Error! Failed to get request token.')

api = tweepy.API(auth)

id_iter = iter(df["tweet_id"])

results = []

for i in trange(len(df)//100):

    ids = list(df["tweet_id"].iloc[i*100:i*100+100])
    response = [
            (t.id, t.created_at, t.text) 
            for t in api.statuses_lookup(id_=ids)]

    results.extend(response)

