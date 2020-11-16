#%%
import pandas as pd
import numpy as np
import pickle
import nltk
import re
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

#%%


class TweetDataset(torch.utils.data.Dataset):

    def __init__(self, tweets, special_characters = ",._´&’%'\":€$£!?#"):
        self.tweets = tweets
        
        self.characters = "abcdefghijklmnopqrstuvwxyz0123456789" + special_characters
        self.unknown = 'U'

        self.tweets['text'] = self.tweets['text'].str.lower()

        regex_remove_user_name = re.compile("^(?:@\S+\s)+")
        regex_remove_hyphen = re.compile("\-")
        regex_remove_specail_characters = re.compile(f"[^a-z0-9\s{special_characters}]+")
        regex_remove_links = re.compile("http\S+")
        regex_remove_whitespace = re.compile("\s+")

        regex_remove = {
            '&amp;' : '&',
            '&lt;' : '<',
            '&gt;' : '>',
            '&quot;' : '"',
            '&apos;' : '\'',
        }

        for regex, r in regex_remove.items():
            self.tweets['text'] = self.tweets['text'].str.replace(regex, r)

        self.tweets['text'] = (self.tweets['text']
            .str.replace(regex_remove_user_name, '')
            .str.replace(regex_remove_hyphen, ' ')
            .str.replace(regex_remove_specail_characters, 'U')
            .str.replace(regex_remove_links, '')
            .str.replace(regex_remove_whitespace, ' ')
            .str.strip()
        )

        self.alphabet = self.characters + self.unknown + ' '

        
        self.unique_bigrams = [x+y for x in self.alphabet for y in self.alphabet]
        
        bigram_mapper = {x : i  for i, x in enumerate(self.unique_bigrams)}
        encoded = []

        for i, text in enumerate(tqdm(self.tweets['text'])):
            encoded.append([bigram_mapper[ a+b] for a, b in zip(text[1:], text[:-1])])

        self.encoded = encoded

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        pass

tweets = pd.read_pickle("../data/interim/200316.pkl")
obj = TweetDataset(tweets)

# %%
class GenerateNgrams():
    def __init__(self, texts, n):
        self.texts = texts
        self.n = n

    def clean_texts(self):
        self.clean_texts = [
            re.sub(
                "[^a-zA-Z0-9\s,.-_´&%'\":€$£!?']",
                "",
                re.sub(" http\S+", "", re.sub("\s", " ", text)),
            )
        .replace(u"\xa0", u" ")
        .lower()
        if isinstance(text, str)
        else ""
        for text in self.texts
        ]   

    def create_n_grams(self):
        

        self.n_grams = [list(nltk.ngrams(clean_text, self.n)) for clean_text in self.clean_texts]

        self.unique_chars = sorted(''.join(set(''.join(self.clean_texts )))) # Assume all characters are in tweets from df1

        self.unique_bigrams = [x+y for x in self.unique_chars for y in self.unique_chars]

        self.bigram_mapper = dict(zip(self.unique_bigrams, range(len(self.unique_bigrams))))


        



# %%
ngrams = GenerateNgrams(
    texts = tweets.text,
    n = 2
)
ngrams.clean_texts()
ngrams.create_n_grams()
# %%
import torch


class TwitterDataset:
    """ Twitter dataset """

    def __init__(self, file_path):
        

    def __len__(self):


    def __getitem__(self, idx):



#%%
# Loading step
def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

items = loadall("../data/processed/serilized_tweets.pkl")
# %%
f = open("../data/processed/serilized_tweets.pkl", "rb")

# %%
