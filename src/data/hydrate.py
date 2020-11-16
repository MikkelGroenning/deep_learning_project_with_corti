
import pandas as pd
import tweepy
from tqdm import trange
import time
import pickle
import json
from typing import List
from pathlib import Path
from argparse import ArgumentParser


def get_api():

    with open(Path(__file__).parent / "credentials.json") as f:
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


def _pickle_object(object, path):

    with open(path, "wb") as f:
        pickle.dump(object, f)


def _unpickle_object(path):

    with open(path, "rb") as f:
        return pickle.load(f)

def _remove_file(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass

def _save_data_frame(results, path):

    hydrated_df = pd.DataFrame.from_records(
        results,
        columns=["tweet_id", "created_at", "text"],
    )
    hydrated_df.to_pickle(path)


class Hydrator:
    def __init__(self) -> None:

        self.api = get_api()
        self.backoff_counter = 1

    def hydrate_tweets_by_date(self, date_string):

        cache_path = Path(f"Data/Interim/hydrated/{date_string}_tmp.pkl")
        cache_every = 100

        non_hydrated_path = Path(f"Data/Interim/Tweet{date_string}.tsv")
        non_hydrated_df = pd.read_csv(non_hydrated_path)

        hydrated_path = Path(f"Data/Interim/hydrated/{date_string}.pkl")
        if hydrated_path.is_file():
            _remove_file(cache_path)
            return

        n = len(non_hydrated_df)

        if cache_path.is_file():
            i_start, results_so_far = _unpickle_object(cache_path)
        else:
            i_start, results_so_far = 0, []

        if i_start >= n // 100 + 1:
            _save_data_frame(results_so_far, hydrated_path)
            _remove_file(cache_path)
            return

        for i in trange(i_start, n // 100 + 1):

            ids = list(non_hydrated_df["tweet_id"].iloc[i * 100 : i * 100 + 100])
            response = self.get_tweets(ids)
            results_so_far.extend(response)

            if i % cache_every == cache_every - 1 or i == n // 100:
                _pickle_object((i + 1, results_so_far), cache_path)

        _save_data_frame(results_so_far, hydrated_path)
        _remove_file(cache_path)

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

                if "Failed to send request" in e.reason:
                    print("Failed to send request, Sleeping for 10 seconds")
                    time.sleep(10)
                else:
                    print(e.reason)
                    print("Sleeping for {} seconds".format(60 * self.backoff_counter))
                    time.sleep(60 * self.backoff_counter)
                    self.backoff_counter += 1

        return response

if __name__ == "__main__":

    # parser = ArgumentParser()
    # parser.add_argument("dates", nargs="+")
    # args = parser.parse_args()

    dates = ["200316"]
    hydrator = Hydrator()
    
    for date_string in dates:
        hydrator.hydrate_tweets_by_date(date_string)
