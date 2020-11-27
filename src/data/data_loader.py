#%%

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from torch.nn.utils.rnn import pack_sequence
import re

special_characters = ",._´&’%'\":€$£!?#"

character_set = {
    "characters": "abcdefghijklmnopqrstuvwxyz0123456789" + special_characters,
    "unknown": "U",
    "end_string": "E",
    "padding_token": "P",
    "space": " ",
    "user": "@",
}

alphabet = "".join(character_set.values())
character_to_number = {x: i for i, x in enumerate(alphabet)}

regex_html_tags = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
}

regex_prefix_user_name = re.compile("^(?:@\S+\s)+")
regex_inner_user_name = re.compile("@\S+")
regex_links = re.compile("http\S+")
regex_whitespace = re.compile("[\s|-]+")
regex_unknown = re.compile(f"[^{alphabet}]+")


class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, tweets):

        self.tweets = tweets

        for pattern_string, char in regex_html_tags.items():
            self.tweets["text"] = self.tweets["text"].str.replace(pattern_string, char)

        self.tweets["text"] = (
            self.tweets["text"]
            .str.lower()
            .str.replace(regex_prefix_user_name, "")
            .str.replace(regex_inner_user_name, character_set["user"])
            .str.replace(regex_links, "")
            .str.replace(regex_whitespace, character_set["space"])
            .str.replace(regex_unknown, character_set["unknown"])
            .str.strip()
        )

        self.tweets["text"] = (
            tweets["text"] + character_set["end_string"]
        )

        encoded = []

        for text in self.tweets["text"]:
            encoded.append(torch.LongTensor([character_to_number[c] for c in text]))

        self.encoded = encoded

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, indices):
        values = [self.encoded[i] for i in indices]
        return pack_sequence(sorted(values, key=lambda x: -len(x)))


def get_loader(dataset, batch_size, pin_memory=False):

    sampler = BatchSampler(
        RandomSampler(dataset), batch_size=batch_size, drop_last=False
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        pin_memory=pin_memory,
    )