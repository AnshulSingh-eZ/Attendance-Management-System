import pickle

with open('main_embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
    print(embeddings)