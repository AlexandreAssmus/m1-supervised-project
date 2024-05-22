### IMPORT #################################################################################################################
from lxml import etree
import os
import torch
from transformers import CamembertTokenizer, CamembertModel
import matplotlib.pyplot as plt

### TEXT EXTRACTION #########################################################################################################

xmlFolderPath="../../examples-ls-fr/xml"

citations=[]
for filename in os.listdir(xmlFolderPath):
    file_path = os.path.join(xmlFolderPath, filename)
    if os.path.isfile(file_path) and filename.endswith('.xml'):
        tree = etree.parse(file_path)
        root = tree.getroot()
        for elmt in tree.iter():
            if elmt.tag.endswith("quote"):
                citation={"text":"","lexies":[]}
                if elmt.text!=None:
                    citation["text"]=elmt.text
                for child in elmt:
                    if child.tag.endswith("seg"):
                        citation["lexies"].append(child.attrib["source"].split('/')[-1])
                        if child.text!=None:    
                            citation["text"]+=child.text
                        if child.tail!=None:    
                            citation["text"]+=child.tail
                citations.append(citation)

# print(len(citations))
# for citation in citations:
#     print(citation["text"])
#     print(citation["lexies"])
#     print("----")

### TOKENIZARION AND UNTOKENIZATION ############################################################################################

def tokenize(corpus,tokenizer):
    """ returns the tokens of the sentences passed sentences as computed by the model """
    tokens=tokenizer(corpus,return_tensors='pt',padding=True)
    return tokens['input_ids']

def untokenize(input_ids,tokenizer):
    output=[]
    for input in input_ids:
        output.append(tokenizer.convert_ids_to_tokens(input))
    return output

### EMBEDDINGS COMPUTATION #####################################################################################################

def computeEmbeddings(tokens,model): 
    output_embeddings = []
    ####### INPUT EMBEDDINGS ###### 
    embedding_layer = model.embeddings
    output_embeddings.append(embedding_layer(tokens))
    for i in range(11):
        encoder_layer = model.encoder.layer[i]
        encoder_embeddings = encoder_layer(hidden_states=output_embeddings[i])[0]
        output_embeddings.append(encoder_embeddings)
    final_embeddings = model.encoder(output_embeddings[0])[0]
    output_embeddings.append(final_embeddings)
    return output_embeddings

### COSINE SIMILARITY ##########################################################################################################

def cosineSimilarity(embeddings1,embeddings2):
    CosSim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    output=CosSim(embeddings1,embeddings2)
    return(output)

def pairCosineSimilarity(embeddings,layer,pos1,pos2):
    CosSim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cs=CosSim(embeddings[layer][0][pos1],embeddings[layer][0][pos2])
    output = cs.item()
    return(output)

def storeCosineSimilarity(embeddings,pos1,pos2):
    cosine_similarities = []
    for layer in range(13):
        cosine_similarities.append(pairCosineSimilarity(embeddings,layer,pos1,pos2))
    return cosine_similarities

### PLOTTING ####################################################################################################################

def cosineSimilarityPlot(embeddings,pos1,pos2):
    cosine_similarities = storeCosineSimilarity(embeddings,pos1,pos2)
    plt.figure(figsize=(6, 4))
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between Embeddings per layer')
    plt.plot(cosine_similarities)
    plt.xticks(range(13))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

### MAIN ###########################################################################################################################

def main():

    # model definition
    model_name = "camembert-base"
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertModel.from_pretrained(model_name)

    # corpus definition
    corpus1 = ["Il mangeait une glace à la fraise en se contemplant dans la glace."]
    corpus2 = ["Elle mange une glace à la fraise et lui une glace au chocolat."]

    # position
    tokens1 = tokenize(corpus1,tokenizer)
    untokenized1 = untokenize(tokens1,tokenizer)
    print(untokenized1)
    pos11 = 5
    pos12 = 16
    print(untokenized1[0][pos11],untokenized1[0][pos12])

    tokens2 = tokenize(corpus2,tokenizer)
    untokenized2 = untokenize(tokens2,tokenizer)
    print(untokenized2)
    pos21 = 4
    pos22 = 12
    print(untokenized2[0][pos21],untokenized2[0][pos22])

    # experiment
    embeddings1 = computeEmbeddings(tokens1,model)
    cosineSimilarityPlot(embeddings1,pos11,pos12)

    embeddings2 = computeEmbeddings(tokens1,model)
    cosineSimilarityPlot(embeddings2,pos21,pos22)
    


if __name__ == "__main__":
    main()