### IMPORT #################################################################################################################
from lxml import etree
import os
import torch
from transformers import CamembertTokenizer, CamembertModel

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

def tokenize(sentences,tokenizer):
    """ returns the tokens of the sentences passed sentences as computed by the model """
    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
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
    for i in range(10):
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

def cosinePairSimilarity(embeddings1,embeddings2,layer,pos1,pos2):
    CosSim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cs=CosSim(embeddings1[layer][0][pos1],embeddings2[layer][0][pos2])
    output = cs.item()
    return(output)

### PLOTTING ####################################################################################################################

def cosineSimilarityPlot(embedding1,embedding2):
    pass

### MAIN ###########################################################################################################################

def main():

    # model definition
    model_name = "camembert-base"
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertModel.from_pretrained(model_name)

    # corpus definition
    corpus = ["Il a mangé une énorme glace."]
    corpus2 = ["Il se regarde dans la glace."]

    # experiment
    tokens = tokenize(corpus,tokenizer)
    e = computeEmbeddings(tokens,model)
    print(type(e[2]))
    print(e[2].size())
    cs = cosineSimilarity(e[2][0][2],e[2][0][4])
    print(type(cs))
    print(cs.item())


if __name__ == "__main__":
    main()