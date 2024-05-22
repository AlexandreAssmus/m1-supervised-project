import torch
from transformers import CamembertTokenizer, CamembertModel,CamembertForMaskedLM






def tokenize(sentences,tokenizer):
    """ returns the tokens of the sentences passed sentences as computed by the model """
    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    return tokens['input_ids']

def untokenize(input_ids,tokenizer):
    output=[]
    for input in input_ids:
        output.append(tokenizer.convert_ids_to_tokens(input))
    return output


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



def cosineSimilarity(embeddings1,embeddings2):
    CosSim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    output=CosSim(embeddings1,embeddings2)
    return(output.item())

def wordToToken(word,tokenizer):
    t=tokenize([word],tokenizer)
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

def maskPrediction(sentences,tokenizer,model):
    prediction=[]
    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    input_ids=tokens["input_ids"]
    model_output = model(**tokens,output_hidden_states=True)
    prob=model_output.logits.softmax(-1)
    for sentence_index,sentence_prob in  enumerate(prob):
        mask_token_index=input_ids[sentence_index].tolist().index(tokenizer.mask_token_id)
        mask_token_prob=sentence_prob[mask_token_index]
        top_token_prob,top_token_id=mask_token_prob.topk(1,-1)
        predicted_token_str=untokenize([[top_token_id]],tokenizer)
        prediction.append(predicted_token_str[0][0])
    return prediction

def maskPredictionWithEmbeddings(sentences,tokenizer,model):
    prediction=[]
    output_embeddings=[]
    
    expected_tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    expected_output=model(**expected_tokens,output_hidden_states=True)


    tokens=tokenizer(sentences,return_tensors='pt',padding=True)
    input_ids=tokens["input_ids"]
    input_embeddings = model.roberta.embeddings(input_ids)
    roberta_output=model.roberta.encoder(input_embeddings)
    logit_output=model.lm_head(roberta_output['last_hidden_state'])
    print(logit_output)
    print(expected_output.logits)
    prob=logit_output.softmax(-1)
    for sentence_index,sentence_prob in  enumerate(prob):
        mask_token_index=input_ids[sentence_index].tolist().index(tokenizer.mask_token_id)
        mask_token_prob=sentence_prob[mask_token_index]
        top_token_prob,top_token_id=mask_token_prob.topk(1,-1)
        predicted_token_str=untokenize([[top_token_id]],tokenizer)
        prediction.append(predicted_token_str[0][0])
    return prediction


def attentionMatrices(sentences,tokenizer,model):
    encoded_input = tokenizer(sentences, return_tensors='pt')
    outputs = model(**encoded_input)
    return outputs.attentions 


modelName="camembert-base"

tokenizer = CamembertTokenizer.from_pretrained(modelName)
maskedModel = CamembertForMaskedLM.from_pretrained(modelName)
model=CamembertModel.from_pretrained(modelName, output_attentions=True)

inputSentences=["Elle se regarde dans la glace. Il se contemple dans le miroir de sa chambre."]
tokenStr=untokenize(tokenize(inputSentences,tokenizer),tokenizer)
print(tokenStr)
wordsIndexes=[(6,15),(3,18)]
for sentence_index in range(len(inputSentences)):
    for wordI in wordsIndexes:
        print(tokenStr[sentence_index][wordI[0]],tokenStr[sentence_index][wordI[1]])
attentions = attentionMatrices(inputSentences, tokenizer, model)
print(len(attentions))
print(attentions[0].shape)
for layer in range(12):
    for head in range(12):   
        mean=attentions[layer][0][head].mean()
        print(f"l: {layer}, h: {head}")
        for wordI in wordsIndexes:
            val1=attentions[layer][0][head][wordI[0]][wordI[1]]/mean
            val2=attentions[layer][0][head][wordI[1]][wordI[0]]/mean
            print(f"{val1} , {val2}")
#print(attentions[0][0][0])
exit()



inputSentences=["Vous savez où est la <mask> la plus proche?",
    "La Seine est un <mask>.",
    "Je cherche urgemment un endroit où retirer de l'<mask>.",
    "Je mange une <mask> à la fraise.",
    "Maman se regarde dans la <mask> pour se coiffer."]
# print(maskPrediction(inputSentences,tokenizer,maskedModel))
print(maskPredictionWithEmbeddings(inputSentences,tokenizer,maskedModel))


inputSentences=["elle mange une bonne glace","il dévore sa grande glace","un reflet dans la glace"]
inputLexie=["glace I","glace I", "glace II" ]
wordsToAnalyse=["glace","sorbet","miroir"]

input_ids=tokenize(inputSentences,tokenizer)
print(input_ids)
print(untokenize(input_ids,tokenizer))
embeddings_array=computeEmbeddings(input_ids,model)

output=analyseToken(wordToToken("glace",tokenizer),input_ids,embeddings_array, inputLexie)
for out in output:
    print("----step----")
    print(out)




