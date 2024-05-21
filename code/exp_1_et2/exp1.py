

# exp 1 - fil conducteur
#TODO
# calculer moyenne attention relative dans le corpus selon les couches et têtes (à voir)
# extraire fn lex pour trouver les mots liés par elles
# faire une étude stats pour choisir quelle fn lex eventuellement
# idem pour trouver un seuil, moyenne la meilleure solution?
# extraire attention relative pour paire de mots reliés par dite fn lex
# comparer cette attention relative avec moyenne calculée
# prender en compte la repr dans l'espace ----> gare au hors sujets


### IMPORTS ###################################################
import torch
from transformers import CamembertTokenizer, CamembertModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


### IMPACT DU POSITIONAL ENCODING ############################


### CHOIX DE LA FONCTION LEXICALE ############################

path = "/home/marina/Documents/M1/facS8/supproj/m1-supervised-project/lexical-system-fr/ls-fr-V3/15-lslf-rel.csv"
df = pd.read_csv(path, delimiter='\t')
lexfn_count = df["lf"].value_counts()
print(lexfn_count.head())


### RECUPERATION DES MATRICES D'ATTENTION #####################

modelName="camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(modelName)
model=CamembertModel.from_pretrained(modelName, output_attentions=True)

nb_layers = model.config.num_hidden_layers
nb_heads = model.config.num_attention_heads



def tokenize(sentences,tokenizer):
    """ returns the tokens of the sentences passed sentences as computed by the model """
    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    return tokens['input_ids'] # torch.tensor

def untokenize(input_ids,tokenizer):
    output=[]
    for input in input_ids:
        output.append(tokenizer.convert_ids_to_tokens(input))
    return output


def attentionMatrices(sentences,tokenizer,model):
    encoded_input = tokenizer(sentences, return_tensors='pt')
    outputs = model(**encoded_input)
    return outputs.attentions


def layerAttentionMatrices(sentences,tokenizer,model,layer):
    encoded_input = tokenizer(sentences, return_tensors='pt')
    outputs = model(**encoded_input)
    output = outputs.attentions
    return output[layer]


### RECUPERATION DE L'ATTENTION RELATIVE ENTRE DEUX MOTS DONNÉS ###


def layerRelativeAttention(corpus, layer,  pos1, pos2, tokenizer, model):
    layer_attention_matrix = layerAttentionMatrices(corpus, tokenizer, model, layer)
    layer_relative_attention = []
    for head in range(nb_heads):
        layer_relative_attention.append(layer_attention_matrix[0][head][pos1][pos2])
    return [tensor.item() for tensor in layer_relative_attention]

def relativeAttention(corpus, pos1, pos2, tokenizer, model):
    relative_attention = []
    for layer in range(nb_layers):
        relative_attention.append(layerRelativeAttention(corpus, layer, pos1,pos2,tokenizer,model))
    return relative_attention



### PREMIERS TRACÉS ##################################################


def layerPairPlot(corpus,pos1,pos2, tokenizer, model):
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink', 'grey']
    relative_attentions = relativeAttention(corpus,pos1,pos2, tokenizer, model)
    for i in range(12):
        plt.plot(relative_attentions[i], marker='o',linestyle='-',color=colors[i],label=f'Layer{i}')
    plt.legend()
    plt.show()



### NORMALISATION AVEC MOYENNE LOCALE ##############################


### CALCUL MOYENNE GLOBALE ########################################

def headAverageAttention(corpus, layer, head, tokenizer, model):
    attention_matrices = attentionMatrices(corpus,tokenizer,model)
    average_attention = torch.mean(attention_matrices[layer][0][head])
    return(average_attention.item())


def layerAverageAttention(corpus, layer, tokenizer, model):
    head_averages = []
    for head in range(nb_heads):
        head_averages.append(headAverageAttention(corpus,layer,head, tokenizer,model))
    array = np.array(head_averages)
    layer_average = np.mean(array)
    return layer_average




### NORMALISATION AVEC MOYENNE GLOBALE ############################

### POSITIONNEMENT PAR RAPPORT AUX MOYENNES DE L'ATTENTION #########


### ETUDE STATISTIQUE DE L'ATTENTION ##############################


### GESTION DES UNITÉS LEXICALES À PLUSIEURS TOKENS ###############


### MAIN #########################################################

def main():

    model_name = "camembert-base"
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertModel.from_pretrained(model_name,output_attentions=True)


    corpus = ["Aujourd'hui est une belle journée."]


    # tokens = tokenize(corpus, tokenizer)
    # layer0 = layerAttentionMatrices(corpus,tokenizer,model,0)
    # attention_matrices = attentionMatrices(corpus,tokenizer,model)
    # # print(untokenize(tokens,tokenizer))
    # # print("nb of tokens:", len(tokens[0]))
    # # print(layer0.shape)
    # # print(len(attention_matrices))
    # # print(attention_matrices[0].shape)

    # relative_attentions = relativeAttention(corpus, 0, 1, tokenizer, model)
    # #print(relative_attentions)

    # layerPairPlot(corpus,0,1,tokenizer,model)

    # matrices = attentionMatrices(corpus,tokenizer,model)
    # print(type(matrices))

    # #globalAverageAttention(corpus,tokenizer,model)


    attention_matrices = attentionMatrices(corpus,tokenizer,model)
    print(type(attention_matrices)) # tuple
    print(len(attention_matrices)) #12
    print(len(attention_matrices[0])) #1
    print(len(attention_matrices[0][0])) #12
    print(attention_matrices[0][0])
    print(len(attention_matrices[0][0][0])) #10
    print(len(attention_matrices[0][0][0][0])) #10

    print(attention_matrices[0][0][0])
    average_attention = torch.mean(attention_matrices[0][0][0])
    print(average_attention.item())
    print(headAverageAttention(corpus,0,0,tokenizer,model))



if __name__ == "__main__":
    main()