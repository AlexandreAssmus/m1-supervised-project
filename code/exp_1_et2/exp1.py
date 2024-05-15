

# exp 1 - fil conducteur
#TODO
# calculer moyenne attention relative dans le corpus selon les couches et têtes (à voir)
# extraire fn lex pour trouver les mots liés par elles
# faire une étude stats pour choisir quelle fn lex eventuellement
# idem pour trouver un seuil, moyenne la meilleure solution?
# extraire attention relative pour paire de mots reliés par dite fn lex
# comparer cette attention relative avec moyenne calculée
# prender en compte la repr dans l'espace ----> gare au hors sujets


### IMPACT DU POSITIONAL ENCODING ############################

import torch
from transformers import CamembertTokenizer, CamembertModel

model_name = "camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(model_name)
model = CamembertModel.from_pretrained(model_name)

v1, v2 = torch.tensor([1,2,3,4,5]), torch.tensor([1,2,3,4,5])
print(cosineSimilarity(v1,v2))


### CHOIX DE LA FONCTION LEXICALE ############################
import pandas as pd

path = "/home/marina/Documents/M1/facS8/supproj/m1-supervised-project/lexical-system-fr/ls-fr-V3/15-lslf-rel.csv"
df = pd.read_csv(path, delimiter='\t')
lexfn_count = df["lf"].value_counts()
print(lexfn_count.head())























# model definition
model = ""
tokenizer = ""


def relative_attention(corpus,word):
    pass
