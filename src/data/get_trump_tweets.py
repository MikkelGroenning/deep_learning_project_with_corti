# importing the module 
import pandas as pd
import numpy as np
import tweepy
import time
import re
from tqdm import trange
from src.data.hydrate import get_api,_pickle_object, _unpickle_object
import torch
from tqdm import tqdm
from pathlib import Path



def hydrate_tweets(tweet_ids, filepath, api):

    exception_list = []
    results = []
    backoff_counter = 1
    # Plus with 1 to get the remainder of division with 100
    for i in trange( (len(tweet_ids) // 100 + 1)):

        while True:
            try:
                
                ids = list(tweet_ids[i * 100 : i * 100 + 100])
                for t in api.statuses_lookup(id_=ids, tweet_mode="extended"):

                    if hasattr(t, "retweeted_status"):
                        full_text = t.retweeted_status.full_text
                        try:
                            retweet = re.findall(r"^RT @([^:]+):", t.full_text)[0]
                        except:
                            exception_list.append(t.id)
                            continue

                    else:  # Not a Retweet
                        full_text = t.full_text
                        retweet = False

                    response = (
                        t.user.id,
                        t.user.name,
                        t.id,
                        t.created_at,
                        full_text,
                        retweet,
                        t.retweet_count,
                        t.favorite_count,
                        t.in_reply_to_status_id,
                        t.in_reply_to_user_id,
                    )

                    results.append(response)

            except tweepy.TweepError as e:
                print(e.reason)
                print("Sleeping for {} seconds".format(60 * backoff_counter))
                time.sleep(60 * backoff_counter)
                backoff_counter += 1

            else:
                break

    tweets_trump = pd.DataFrame(
        results,
        columns=[
            "user_id",
            "user_name",
            "id",
            "created_at",
            "text",
            "retweet",
            "retweet_count",
            "favorite_count",
            "in_reply_to_status_id",
            "in_reply_to_user_id",
        ],
    )
    

    # Filter
    tweets_trump = tweets_trump[tweets_trump['created_at']<= '2019-12-31']

    print(f'Writing {tweets_trump} tweets to lenlocal file')
    tweets_trump.to_pickle(filepath)

  
def get_trump_tweet_ids(df_trump, filepath):
    """
    Extract the tweet id from a dataframe downloaded at trump twitter archive.
    """
    # Remove rows with missing values
    df_trump = df_trump[~df_trump.isna().any(axis=1)]

    # Extract the tweets ids and convert them to integers
    trump_ids = list(df_trump.id_str.astype(int).values)

    with open(filepath, 'w') as output:
        for row in trump_ids:
            output.write(str(row) + '\n')

        print(f'{len(trump_ids)} tweet ids saved.')




character_set = {
    "characters": "abcdefghijklmnopqrstuvwxyz0123456789 ",
    "end_string": "<EOS>",
    "space": " ",
}

regex_html_tags = {
    "&amp;": "and",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
}

regex_prefix_user_name = re.compile("^(?:@\S+\s)+")
regex_inner_user_name = re.compile("@\S+")
regex_links = re.compile("http\S+")
regex_whitespace = re.compile("[\s|-|-|_]+")
regex_unknown = re.compile(f"[^{ character_set['characters'] }]+")


class Embedder:
    def __init__(self, embedder):
        self.embedder = embedder

        rsum = np.zeros(300)
        for word, embedding in embedder.vocab.items():
            rsum += embedder[word]

        self.average = rsum / len(embedder.vocab)

    def embed_tweets_from_trump(self):
        path_hydrated = Path(f"data/interim/hydrated/trump.pkl")
        path_processed = Path(f"data/interim/hydrated/trump_text_processed.pkl")
        path_embedded = Path(f"data/processed/trump_embedding.pkl")

        try:
            tweets = _unpickle_object(path_hydrated)
        except Exception as e:
            print(e)
            print("Hydrated file not")
            raise

        for pattern_string, char in regex_html_tags.items():
            tweets["text_processed"] = tweets["text"].str.replace(pattern_string, char)

        tweets["text_processed"] = tweets["text_processed"] = (
            tweets["text_processed"]
            .str.lower()
            .str.replace(regex_whitespace, character_set["space"])
            .str.strip()
            .str.replace(regex_prefix_user_name, "")
            .str.strip()
            .str.replace(regex_inner_user_name, "")
            .str.strip()
            .str.replace(regex_links, "")
            .str.strip()
            .str.replace(regex_unknown, "")
            .str.strip()
        )

        # Drop empty tweets
        tweets = tweets[tweets.text_processed != '']

        # Only get same columns
        tweets = tweets[['id', 'created_at', 'text', 'text_processed']]

        print('save process tweets')
        # Save the processed data frame
        tweets.to_pickle(path_processed)

        tweets_embedding = []
        for tweet in tweets["text_processed"]:

            embedding_tweet = torch.from_numpy(
                np.vstack(
                    [
                        self.embedder[word]
                        if word in self.embedder.vocab.keys()
                        else self.average
                        for word in tweet.split() + ['EOS']
                    ],
                ),
            )
            embedding_tweet = embedding_tweet.type(torch.float)
            tweets_embedding.append(embedding_tweet)


        torch.save(tweets_embedding, path_embedded)
        print('Finished')


if __name__ == "__main__":
    print('')
    print('__file__:    ', __file__)

    try:
        trump = np.loadtxt("data/raw/trump/trump_id.txt", dtype=int)

    except IOError:
        df_trump_tweets1 = pd.read_csv('data/raw/trump/trump_tweets_1st.csv')  
        df_trump_tweets2 = pd.read_csv('data/raw/trump/trump_tweets_2nd.csv')
        df_trump = pd.concat([df_trump_tweets1, df_trump_tweets2])

        filepath_to_trump = "data/raw/trump/trump_id.txt"
        get_trump_tweet_ids(df_trump, filepath_to_trump)

        trump = np.loadtxt("data/raw/trump/trump_id.txt", dtype=int)

 
    filepath = "data/interim/hydrated/trump.pkl"
    
    
    api = get_api()

    hydrate_tweets(
        tweet_ids=trump,
        filepath=filepath,
        api = api
    )

    # Get embeddings
    word2vec = _unpickle_object("data/processed/Embedding.pickle")
    embedder = Embedder(word2vec)

    embedder.embed_tweets_from_trump()

    print('Success!')