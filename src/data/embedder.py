import pickle
import gensim
import pandas as pd
import numpy as np
import re
import torch
from tqdm import tqdm
from pathlib import Path
from hydrate import _pickle_object, _unpickle_object
from src.data.common import data_train, data_validation, data_test, data_sampled


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

    def embed_tweets_by_date(self, date_string):


        path_embedded = Path(f"data/processed/{date_string}_embedding.pkl")
        tweets = data_sampled.copy()

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



        # print('save process tweets')
        # # Save the processed data frame
        # tweets.to_pickle(path_processed)

        tweets_embedding = []

        for tweet in tqdm(tweets["text_processed"], total=len(tweets)):

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

        embedded_series = pd.Series(tweets_embedding)

        torch.save({
            "train" : embedded_series.loc[embedded_series.index.intersection(data_train.index)].tolist(),
            "validation" : embedded_series.loc[embedded_series.index.intersection(data_validation.index)].tolist(),
            "test" : embedded_series.loc[embedded_series.index.intersection(data_test.index)].tolist()
        }, path_embedded)
        print('Finished')

if __name__ == "__main__":


    dates = ["200316"]
    word2vec = _unpickle_object("data/processed/Embedding.pickle")
    embedder = Embedder(word2vec)

    for date_string in dates:
        embedder.embed_tweets_by_date(date_string)