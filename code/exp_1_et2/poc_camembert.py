import torch
from transformers import CamembertTokenizer, CamembertModel,CamembertForMaskedLM


modelName="camembert-base"


def tokenize(sentences):
    """ returns the tokens of the sentences passed sentences as computed by the model """

    tokenizer = CamembertTokenizer.from_pretrained(modelName)
    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    return tokens['input_ids']

def untokenize(input_ids):
    output=[]
    tokenizer = CamembertTokenizer.from_pretrained(modelName)
    for input in input_ids:
        output.append(tokenizer.convert_ids_to_tokens(input))
    return output


def computeEmbeddings(tokens): 
    """ returns input embedding of arguments tokens"""
    
    output_embeddings = []
    #load model 
    model = CamembertModel.from_pretrained(modelName)

    #print(model)
   
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





def cosineSimilarity(embeddings1,embeddings2):
    CosSim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    output=CosSim(embeddings1,embeddings2)
    return(output.item())

def wordToToken(word):
    t=tokenize([word])
    return(t[0][1].item())


def analyseToken(token,input_ids,embeddings_array,inputLexie):
    output = []
    nb_sentences = input_ids.shape[0]
    for embeddings in embeddings_array:
        step_output = [[],[]]
        for i in range(nb_sentences):
            input1 = input_ids[i]
            if token in input1: 
                index1=input1.tolist().index(token)
                emb1=embeddings[i][index1]
                for j in range(i+1,nb_sentences):
                    input2 = input_ids[j]
                    if token in input2:
                        index2=input2.tolist().index(token)
                        emb2=embeddings[j][index2]
                        dist=cosineSimilarity(emb1,emb2)
                        if inputLexie[i]==inputLexie[j]:
                            step_output[0].append(dist)
                        else:
                            step_output[1].append(dist)
        output.append(step_output)
    return output                   

def maskPrediction(sentences):
    prediction=[]
    tokenizer = CamembertTokenizer.from_pretrained(modelName)
    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    input_ids=tokens["input_ids"]
    model = CamembertForMaskedLM.from_pretrained(modelName)
    model_output = model(**tokens,output_hidden_states=True)
    prob=model_output.logits.softmax(-1)
    for sentence_index,sentence_prob in  enumerate(prob):
        mask_token_index=input_ids[sentence_index].tolist().index(tokenizer.mask_token_id)
        mask_token_prob=sentence_prob[mask_token_index]
        top_token_prob,top_token_id=mask_token_prob.topk(1,-1)
        predicted_token_str=untokenize([[top_token_id]])
        prediction.append(predicted_token_str[0][0])
    return prediction

inputSentences=["Vous savez où est la <mask> la plus proche?",
    "La Seine est un <mask>.",
    "Je cherche urgemment un endroit où retirer de l'<mask>.",
    "Je mange une <mask> à la fraise.",
    "Maman se regarde dans la <mask> pour se coiffer."]
print(maskPrediction(inputSentences))
exit()


inputSentences=["elle mange une bonne glace","il dévore sa grande glace","un reflet dans la glace"]
inputLexie=["glace I","glace I", "glace II" ]
wordsToAnalyse=["glace","sorbet","miroir"]

input_ids=tokenize(inputSentences)
print(input_ids)
print(input_ids)
print(untokenize(input_ids))
embeddings_array=computeEmbeddings(input_ids)
#print(result.shape)
output=analyseToken(wordToToken("glace"),input_ids,embeddings_array, inputLexie)
for out in output:
    print("----step----")
    print(out)




