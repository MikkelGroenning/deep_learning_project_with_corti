import pickle
import gensim
import pandas as pd
import numpy as np
import re
import torch
from tqdm import tqdm
from pathlib import Path
from src.data.hydrate import _pickle_object, _unpickle_object
from src.data.common import data_train, data_validation, data_test, data_sampled, data_trump

if __name__ == "__main__":

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

    word2vec = _unpickle_object("data/processed/Embedding.pickle")
    # word2vec = Embedder(word2vec)

    rsum = np.zeros(300)
    for word, embedding in word2vec.vocab.items():
        rsum += word2vec[word]
    average = rsum / len(word2vec.vocab)

        # def embed_tweets_by_date(self, date_string):

# %% Embed COVID twitter data
    
    path_embedded = Path(f"data/processed/200316_embedding.pkl")
    tweets = data_sampled.copy()

    for pattern_string, char in regex_html_tags.items():
        tweets["text_processed"] = tweets["text"].str.replace(pattern_string, char)

    tweets["text_processed"] = (
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
    tweets_embedding = []

    for tweet in tqdm(tweets["text_processed"], total=len(tweets)):
        embedding_tweet = torch.from_numpy(
            np.vstack(
                [
                    word2vec[word]
                    if word in word2vec.vocab.keys()
                    else average
                    for word in tweet.split() + ['EOS']
                ],
            ),
        )
        embedding_tweet = embedding_tweet.type(torch.float)
        tweets_embedding.append(embedding_tweet)

    embedded_series = pd.Series(tweets_embedding)

# %% Embed trump data

    path_embedded_trump = Path(f"data/processed/trump_embedding.pkl")
    tweets_trump = data_trump.copy()

    for pattern_string, char in regex_html_tags.items():
        tweets_trump["text_processed"] = tweets_trump["text"].str.replace(pattern_string, char)

    tweets_trump["text_processed"] = (
        tweets_trump["text_processed"]
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
    tweets_trump = tweets_trump[tweets_trump.text_processed != '']
    tweets_embedding_trump = []

    for tweet in tqdm(tweets_trump["text_processed"], total=len(tweets_trump)):
        embedding_tweet = torch.from_numpy(
            np.vstack(
                [
                    word2vec[word]
                    if word in word2vec.vocab.keys()
                    else average
                    for word in tweet.split() + ['EOS']
                ],
            ),
        )
        embedding_tweet = embedding_tweet.type(torch.float)
        tweets_embedding_trump.append(embedding_tweet)

    embedded_series_trump = pd.Series(tweets_embedding_trump)

# %% Save data
    torch.save({
        "train" : embedded_series.loc[embedded_series.index.intersection(data_train.index)].tolist(),
        "validation" : embedded_series.loc[embedded_series.index.intersection(data_validation.index)].tolist(),
        "test" : embedded_series.loc[embedded_series.index.intersection(data_test.index)].tolist(),
        "trump" : embedded_series_trump.tolist(),
    }, path_embedded)
    print('Finished')


# %%
