import collections
from pickle import PickleError
import pandas as pd
from pandas.core.construction import is_empty_data
import tweepy
from tqdm import trange
import time
import pickle
import json
from typing import List


def _pickle_object(object, path):

    with open(path, "wb") as f:
        pickle.dump(object, f)


def _unpickle_object(path):

    with open(path, "rb") as f:
        return pickle.load(f)


# TODO: Fix files and directories


class Hydrator:
    def __init__(self) -> None:
        self.api = get_api()
        self.backoff_counter = 1

    def hydrate_tweets_by_date(self, date_string):

        non_hydrated_df = pd.read_csv(f"Data/Interim/Tweet{date_string}.tsv")

        n = len(non_hydrated_df)

        cache_path = f"Data/Interim/hydrated/{date_string}_tmp.pkl"
        cache_every = 100

        try:
            i_start, results_so_far = _unpickle_object(cache_path)
        except FileNotFoundError:
            i_start, results_so_far = 0, []

        for i in trange(i_start, n // 100 + 1):

            ids = list(non_hydrated_df["tweet_id"].iloc[i * 100 : i * 100 + 100])
            response = self.get_tweets(ids)
            results_so_far.extend(response)

            if i % cache_every == cache_every - 1:
                _pickle_object((i + 1, results_so_far), cache_path)

        hydrated_df = pd.DataFrame.from_records(
            results_so_far,
            columns=["tweet_id", "created_at", "text"],
        )
        hydrated_df.to_pickle(f"Data/Interim/hydrated/{date_string}.pkl")

    def get_tweets(self, ids: List):

        # Try until the request is succesful
        try_again = True
        while try_again:
            try:
                response = [
                    (t.id, t.created_at, t.full_text)
                    for t in self.api.statuses_lookup(id_=ids, tweet_mode="extended")
                ]
                try_again = False
            except tweepy.TweepError as e:
                print(e.reason)

                if "Failed to send request" in e.reason:
                    print("Sleeping for 60 seconds")
                    time.sleep(60)
                else:
                    print("Sleeping for {} seconds".format(60 * self.backoff_counter))
                    time.sleep(60 * self.backoff_counter)
                    self.backoff_counter += 1
                
        return response


def get_api():

    with open("DataPrep/credentials.json") as f:
        credentials = json.load(f)

    auth = tweepy.OAuthHandler(credentials["api_key"], credentials["api_secret_key"])
    auth.set_access_token(
        credentials["access_token"], credentials["access_token_secret"]
    )

    try:
        redirect_url = auth.get_authorization_url()  # noqa
    except tweepy.TweepError:
        print("Error! Failed to get request token.")

    return tweepy.API(auth, wait_on_rate_limit=True)


if __name__ == "__main__":

    dates = ["200316"]

    hydrator = Hydrator()

    for date_string in dates:
        hydrator.hydrate_tweets_by_date(date_string)
