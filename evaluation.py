import pandas as pd
import torch
import numpy as np

from speechbrain.pretrained import SpeakerRecognition

if __name__ == '__main__':
    train_embeddings_mean = pd.read_csv("./embeddings/means/train.csv", sep=',', header=0)
    devel_embeddings = pd.read_csv("./embeddings/devel.csv", sep=',', header=0)

    # print(train_embeddings_mean.values[0, 4:])

    x1 = torch.FloatTensor(list(train_embeddings_mean.values[0, 4:]))
    x2 = torch.FloatTensor(list(devel_embeddings.values[0, 4:]))

    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    score = similarity(x1, x2)

    print(score)
