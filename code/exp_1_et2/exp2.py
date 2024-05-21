from lxml import etree
import os
import torch



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

print(len(citations))
for citation in citations:
    print(citation["text"])
    print(citation["lexies"])
    print("----")


### TOKENISATION ET DETOKENISATION #####################################################################



def tokenize(sentences,tokenizer):
    """ returns the tokens of the sentences passed sentences as computed by the model """
    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    return tokens['input_ids']

def untokenize(input_ids,tokenizer):
    output=[]
    for input in input_ids:
        output.append(tokenizer.convert_ids_to_tokens(input))
    return output


### CALCUL DES EMBEDDINGS ###########################################################################


def computeEmbeddings(tokens,model): 
    """ returns input embedding of arguments tokens"""
    
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



### SIMILARITE COSINUS ####################################################################


def cosineSimilarity(embeddings1,embeddings2):
    CosSim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    output=CosSim(embeddings1,embeddings2)
    return(output.item())


### MAIN ##################################################################################


def main():
    print("hello")


if __name__ == "__main__":
    main()