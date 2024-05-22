

# exp 1 - fil conducteur
#TODO
# calculer moyenne attention relative dans le corpus selon les couches et têtes (à voir)
# extraire fn lex pour trouver les mots liés par elles
# faire une étude stats pour choisir quelle fn lex eventuellement
# idem pour trouver un seuil, moyenne la meilleure solution?
# extraire attention relative pour paire de mots reliés par dite fn lex
# comparer cette attention relative avec moyenne calculée
# prender en compte la repr dans l'espace ----> gare au hors sujets


### IMPORTS ########################################################################################################################################
import torch
from transformers import CamembertTokenizer, CamembertModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### IMPACT DU POSITIONAL ENCODING ##################################################################################################################

### TOKENIZATION AND UNTOKENIZATION ################################################################################################################

def tokenize(corpus,tokenizer):
    """ returns the tokens of the sentences passed sentences as computed by the model """
    tokens=tokenizer(corpus,return_tensors='pt',padding=True)
    return tokens['input_ids'] # torch.tensor

def untokenize(input_ids,tokenizer):
    output=[]
    for input in input_ids:
        output.append(tokenizer.convert_ids_to_tokens(input))
    return output

### EXTRACTION OF ATTENTION MATRICES ###############################################################################################################

def attentionMatrices(corpus,tokenizer,model):
    encoded_input = tokenizer(corpus, return_tensors='pt')
    outputs = model(**encoded_input)
    return outputs.attentions

def layerAttentionMatrices(corpus, layer, tokenizer, model):
    attention_matrices = attentionMatrices(corpus, tokenizer, model)
    return attention_matrices[layer]

def headAttentionMatrix(corpus, layer, head, tokenizer, model):
    layer_attention_matrix = layerAttentionMatrices(corpus, layer, tokenizer, model)
    return layer_attention_matrix[0][head]

### EXTRACTION OF RELATIVE ATTENTION BETWEEN TWO GIVEN WORDS #######################################################################################

def headRelativeAttention(corpus, layer, head, pos1, pos2, tokenizer, model):
    head_attention_matrix = headAttentionMatrix(corpus, layer, head, tokenizer, model)
    return head_attention_matrix[pos1,pos2].item()

def layerRelativeAttention(corpus, layer,  pos1, pos2, tokenizer, model):
    nb_heads = model.config.num_attention_heads
    layer_attention_matrix = layerAttentionMatrices(corpus, layer, tokenizer, model)
    layer_relative_attention = []
    for head in range(nb_heads):
        layer_relative_attention.append(layer_attention_matrix[0][head][pos1][pos2])
    return [tensor.item() for tensor in layer_relative_attention]

def relativeAttention(corpus, pos1, pos2, tokenizer, model):
    nb_layers = model.config.num_hidden_layers
    relative_attention = []
    for layer in range(nb_layers):
        relative_attention.append(layerRelativeAttention(corpus, layer, pos1,pos2,tokenizer,model))
    return relative_attention

### RELATIVE ATTENTION PLOTTING #####################################################################################################################

def relativeAttentionPlot(corpus,pos1,pos2, tokenizer, model):
    nb_layers = model.config.num_hidden_layers
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink', 'grey']
    relative_attention = relativeAttention(corpus,pos1,pos2, tokenizer, model)
    for layer in range(nb_layers):
        plt.plot(relative_attention[layer], marker='o',linestyle='-',color=colors[layer],label=f'Layer{layer}')
    plt.legend()
    plt.show()

def layerRelativeAttentionPlot(corpus,layer,pos1,pos2,tokenizer,model):
    layer_relative_attention = layerRelativeAttention(corpus,layer,pos1,pos2,tokenizer,model)
    plt.plot(layer_relative_attention, marker='o',linestyle='-',color='black')
    plt.legend()
    plt.show()

def relativeAttentionSubplot(corpus,pos1,pos2,tokenizer,model):
    plt.figure(figsize=(15, 10))
    nb_layers = model.config.num_hidden_layers
    relative_attention = relativeAttention(corpus,pos1,pos2,tokenizer,model)
    for layer in range(nb_layers):
        plt.subplot(4,3,layer+1)
        plt.plot(relative_attention[layer])
        plt.title(f"Layer {layer}")
        plt.xticks(range(12))
    plt.tight_layout()
    plt.show() 

### COMPUTATION OF AVERAGE ATTENTION RATES ########################################################################################################

def headAverageAttention(corpus, layer, head, tokenizer, model):
    attention_matrices = attentionMatrices(corpus,tokenizer,model)
    average_attention = torch.mean(attention_matrices[layer][0][head])
    return(average_attention.item()) #float

def layerAverageAttention(corpus, layer, tokenizer, model):
    nb_heads = model.config.num_attention_heads
    head_averages = []
    for head in range(nb_heads):
        head_averages.append(headAverageAttention(corpus,layer,head,tokenizer,model))
    array = np.array(head_averages)
    layer_average = np.mean(array)
    return layer_average

def averageAttention(corpus, tokenizer, model):
    nb_layers = model.config.num_hidden_layers
    layer_averages = []
    for layer in range(nb_layers):
        layer_averages.append(layerAverageAttention(corpus,layer,tokenizer,model))
    array = np.array(layer_averages)
    layer_average = np.mean(array)
    return layer_average

### RELATIVE ATTENTION AVERAGE COMPARISON ########################################################################################################

def relativeAttentionCenteredPlot(corpus,pos1,pos2, tokenizer, model):
    nb_layers = model.config.num_hidden_layers
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink', 'grey']
    relative_attention = relativeAttention(corpus,pos1,pos2, tokenizer, model)
    average_attention = averageAttention(corpus,tokenizer,model)
    for layer in range(nb_layers):
        plt.plot(relative_attention[layer]-average_attention, marker='o',linestyle='-',color=colors[layer],label=f'Layer{layer}')
    plt.legend()
    plt.show()

def layerRelativeAttentionCenteredPlot(corpus,layer,pos1,pos2,tokenizer,model):
    layer_relative_attention = layerRelativeAttention(corpus,layer,pos1,pos2,tokenizer,model)
    layer_average_attention = layerAverageAttention(corpus,layer,tokenizer,model)
    plt.plot(layer_relative_attention-layer_average_attention, marker='o',linestyle='-',color='black')
    plt.legend()
    plt.show()

def relativeAttentionCenteredSubplot(corpus,pos1,pos2,tokenizer,model):
    plt.figure(figsize=(15, 10))
    nb_layers = model.config.num_hidden_layers
    nb_heads = model.config.num_attention_heads
    relative_attention = relativeAttention(corpus,pos1,pos2,tokenizer,model)
    for layer in range(nb_layers):
        head_averages = []
        for head in range(nb_heads):
            head_averages.append(headAverageAttention(corpus, layer, head, tokenizer,model))
        plt.subplot(4,3,layer+1)
        to_plot = [x - y for x, y in zip(relative_attention[layer], head_averages)]
        #plt.plot(relative_attention[layer]-head_averages)
        plt.plot(to_plot)
        plt.title(f"Layer {layer}")
        plt.xticks(range(12))
    plt.tight_layout()
    plt.show() 

### CENTERING #####################################################################################################################################

def headCentering(corpus,layer,head,tokenizer,model):
    head_attention_matrix = headAttentionMatrix(corpus, layer, head, tokenizer,model)
    head_average_attention = headAverageAttention(corpus,layer,head,tokenizer,model)
    centered_matrix = head_attention_matrix-head_average_attention
    return centered_matrix

def layerCentering(corpus,layer,tokenizer,model):
    layer_attention_matrix = layerAttentionMatrices(corpus,layer,tokenizer,model)
    layer_average_attention = layerAverageAttention(corpus,layer,tokenizer,model)
    centered_matrices = layer_attention_matrix-layer_average_attention
    return centered_matrices[0]

def centering(corpus, tokenizer, model):
    nb_layers = model.config.num_hidden_layers
    nb_heads = model.config.num_attention_heads
    attention_matrices = attentionMatrices(corpus,tokenizer,model)
    average_attention = averageAttention(corpus,tokenizer,model)
    centered_matrices = []
    for layer in range(nb_layers):
        layer_centered_matrices = []
        for head in range(nb_heads):
            clean_head_attention_matrix = attention_matrices[layer][0][head].tolist()
            layer_centered_matrices.append(clean_head_attention_matrix-average_attention)
        centered_matrices.append(layer_centered_matrices)
    return centered_matrices # list of lists, 12x12

### PLOTTING CENTERING #############################################################################################################################


### MAIN - EXPERIMENT ##############################################################################################################################

def main():

    # model definition
    model_name = "camembert-base"
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertModel.from_pretrained(model_name,output_attentions=True)

    # corpus definition
    corpus = ["Aujourd'hui est une belle journée."]

    # lexical function selection
    path = "../../lexical-system-fr/ls-fr-V3/15-lslf-rel.csv"
    df = pd.read_csv(path, delimiter='\t')
    lexfn_count = df["lf"].value_counts()
    print(lexfn_count.head())   

    # experiment
    relativeAttentionCenteredPlot(corpus,0,1,tokenizer,model)
    layerRelativeAttentionCenteredPlot(corpus,0,0,1,tokenizer,model)
    relativeAttentionCenteredSubplot(corpus,0,1,tokenizer,model)





if __name__ == "__main__":
    main()