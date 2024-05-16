

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


### CHOIX DE LA FONCTION LEXICALE ############################
import pandas as pd

path = "/home/marina/Documents/M1/facS8/supproj/m1-supervised-project/lexical-system-fr/ls-fr-V3/15-lslf-rel.csv"
df = pd.read_csv(path, delimiter='\t')
lexfn_count = df["lf"].value_counts()
print(lexfn_count.head())


### RECUPERATION DES MATRICES D'ATTENTION #####################


import torch
from transformers import CamembertTokenizer, CamembertModel

modelName="camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(modelName)
model=CamembertModel.from_pretrained(modelName, output_attentions=True)

def tokenize(sentences,tokenizer):
    """ returns the tokens of the sentences passed sentences as computed by the model """
    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    return tokens['input_ids']


def attentionMatrices(sentences,tokenizer,model):
    encoded_input = tokenizer(sentences, return_tensors='pt')
    outputs = model(**encoded_input)
    return outputs.attentions


def LayerAttentionMatrices(sentences,tokenizer,model,layer):
    encoded_input = tokenizer(sentences, return_tensors='pt')
    outputs = model(**encoded_input)
    return outputs.attentions[layer]



corpus = ["Aujourd'hui est une belle journée."]
tokens = tokenize(corpus, tokenizer)
layer0 = LayerAttentionMatrices(corpus,tokenizer,model,0)
attention_matrices = attentionMatrices(corpus,tokenizer,model)


print(len(tokens))
print(layer0.shape)
print(len(attention_matrices))
print(attention_matrices.head())






### ETUDE STATISTIQUE DE L'ATTENTION ##############################


















# model definition
model = ""
tokenizer = ""


def relative_attention(corpus,word):
    pass
