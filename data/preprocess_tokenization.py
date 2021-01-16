# This file is used to process the data into wordpiece tokens that are used in the bert model. The annotations are also
# adjusted according to the new indexing of the tokens in a sentence
from transformers import *
from constants import *
import json
from tqdm import tqdm
paths = [TRN_PATH, DEV_PATH, TST_PATH]
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# We need to know what is the difference between the tokenizing raw text and tokenizing the series of tokens.
# We prefer the second method because of the annotation

# test BERT tokenizer
encoded = tokenizer.encode("Here is some text to encode", add_special_tokens=True)
# print(encoded)
# print(tokenizer.convert_ids_to_tokens(encoded))

# test the input files
for j in [0, 1, 2]:
    file_path = paths[j]
    file_names = ["trn_bert_base_cased.json", "dev_bert_base_cased.json", "tst_bert_base_cased.json"]
    speakers = []
    entities = []

    with open(file_path, "r") as f:
        data = json.load(f)
    print(data["season_id"])
    print("number of episodes:", len(data["episodes"]))
    for epi in tqdm(data["episodes"]):
        for scene in epi["scenes"]:
            for utterance in scene["utterances"]:

                raw_text = utterance["transcript"]
                tokenized_text = utterance["tokens"]
                character_entities = utterance["character_entities"]

                token_mapping_bert_base_cased = []
                tokens_id_bert_base_cased = []
                tokens_bert_base_cased = []
                character_entities_bert_base_cased = []
                for sentence in tokenized_text:
                    sentence_old2new_mapping = []
                    sentence_tokens = []
                    prior_tokens = 0
                    for token in sentence:
                        sentence_old2new_mapping.append(prior_tokens)
                        word_piece_token = tokenizer.encode(token, add_special_tokens=False)

                        prior_tokens += len(word_piece_token)
                        sentence_tokens.extend(word_piece_token)

                    sentence_old2new_mapping.append(prior_tokens)
                    token_mapping_bert_base_cased.append(sentence_old2new_mapping)
                    tokens_id_bert_base_cased.append(sentence_tokens)
                    tokens_bert_base_cased.append(tokenizer.convert_ids_to_tokens(sentence_tokens))

                # print(token_mapping_bert_base_cased)
                # print(tokens_bert_base_cased)
                # print(tokens_id_bert_base_cased)

                character_entities_bert_base_cased = []
                for i, sentence in enumerate(character_entities):
                    if len(sentence) == 0:
                        character_entities_bert_base_cased.append([])
                    else:
                        mapping = token_mapping_bert_base_cased[i]
                        try:
                            entity_list = [[mapping[triplet[0]], mapping[triplet[1]]] + triplet[2:] for triplet in sentence]
                        except:
                            print(mapping)
                            print(sentence)
                        character_entities_bert_base_cased.append(entity_list)

                # print(character_entities_bert_base_cased)

                # save the tokens, the token mappings, and the new annotations to the data
                utterance["token_mapping_bert_base_cased"] = token_mapping_bert_base_cased
                utterance["tokens_bert_base_cased"] = tokens_bert_base_cased
                utterance["tokens_id_bert_base_cased"] = tokens_id_bert_base_cased
                utterance["character_entities_bert_base_cased"] = character_entities_bert_base_cased

    with open(file_names[j], "w") as fout:
        json.dump(data, fout)

