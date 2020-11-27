import gensim
import pickle
import os

# Set path to current wd
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

fname_open = '../Data/Raw/wiki-news-300d-1M.vec'
fname_write = '../Data/Processed/Embedding.pickle'

# Load model
data = gensim.models.KeyedVectors.load_word2vec_format('../Data/Raw/wiki-news-300d-1M.vec')

# Write model to pickle
with open(fname_write, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)