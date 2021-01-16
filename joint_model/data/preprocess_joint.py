# This file is used for the preprocess the data for c2f models.
# This will include the tokenzier for 4 models, BERT-base-cased, BERT-large-cased, SpanBERT base/large cased
# https://huggingface.co/SpanBERT/spanbert-large-cased
# https://huggingface.co/SpanBERT/spanbert-base-cased
import torch
from transformers import *
import json
from tqdm import tqdm
from collections import Counter

MODELS = [(BertModel,       BertTokenizer,       'bert-base-cased'),
          (BertModel,  BertTokenizer,  'bert-large-cased'),
          (AutoModel,       AutoTokenizer,       "SpanBERT/spanbert-base-cased"),
          (AutoModel,       AutoTokenizer,       "SpanBERT/spanbert-large-cased")]

data_folder = "../../data/"
data_path = [data_folder + "character-identification-trn.json", data_folder + "character-identification-dev.json",
             data_folder + "character-identification-tst.json"]
data_name = ["trn", "dev", "tst"]

character_name_to_id = {
    "Ross Geller":0,
    "Rachel Green":1,
    "Chandler Bing":2,
    "Monica Geller":3,
    "Joey Tribbiani":4,
    "Phoebe Buffay":5,
    "Emily": 6,
    "Richard Burke": 7,
    "Carol Willick":8,
    "Ben Geller": 9,
    "Peter Becker":10,
    "Judy Geller":11,
    "Barry Farber":12,
    "Jack Geller": 13,
    "Kate Miller":14,
    "#OTHER#":15,
    "#GENERAL#": 16
    }

character_id_to_name = ["Ross Geller",
    "Rachel Green",
    "Chandler Bing",
    "Monica Geller",
    "Joey Tribbiani",
    "Phoebe Buffay",
    "Emily",
    "Richard Burke",
    "Carol Willick",
    "Ben Geller",
    "Peter Becker",
    "Judy Geller",
    "Barry Farber",
    "Jack Geller",
    "Kate Miller",
    "#OTHER#",
    "#GENERAL#"]

max_seq_len = 512
dummy_mention_id = "epsilon"
# This file will but the script of each scene as a document, and use Independent setting to cut the document into
# segments of length 512 tokens. Then will extract mentions with their position and mention IDs
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)


    speaker_name_to_ids = {"": 0}
    speaker_id_to_names = [""]
    # process the speaker information, count all speakers 
    print("process speakers")
    for i in range(3):
        data_file_path = data_path[i]
        with open(data_file_path, "r") as fin:
            data = json.load(fin)
        for epi in tqdm(data["episodes"]):
            for scene in epi["scenes"]:
                for utterance in scene["utterances"]:
                    joint_speakers = "_".join(utterance["speakers"])
                    if joint_speakers in speaker_name_to_ids:
                        continue
                    else:
                        speaker_name_to_ids[joint_speakers] = len(speaker_id_to_names)
                        speaker_id_to_names.append(joint_speakers)


    # Process other information
    print("process the rest")
    for i in range(3):
        data_file_path = data_path[i]
        batch_sizes = []
        number_of_mentions = []
        number_of_links = []

        with open(data_file_path, "r") as fin:
            data = json.load(fin)

        modified_pretrained_weights = "".join([s for s in pretrained_weights if s != "/"])
        print(modified_pretrained_weights)
        with open(data_name[i]+"_"+modified_pretrained_weights+"_batches.json", "w") as fout:

            all_mention_count = 0
            all_mention_ids = []
            for epi in tqdm(data["episodes"]):

                for scene in epi["scenes"]:
                    # all tokens in the scene
                    # If a 512 segment cannot include a new sentence
                    # The new sentence will be put into a new segment
                    all_scene_tokens = [[]]
                    current_segment_count = 0
                    # The meaningful token masks are 1s, and PAD are 0s
                    all_scene_token_masks = [[]]
                    # The speaker ids 
                    all_scene_speaker_ids = [[]]
                    # The speaker names 
                    all_scene_speaker_names = [[]]

                    # [mention_id, number of segment, start position in segment, end position in segment]
                    mention_id_with_positions = []
                    mentionid2positions = {}

                    # [character_id_of_each_mention]
                    mention_character_id = []
                    # [character_name_of_each_mention]
                    mention_character_name = []
                    # The links for the mentions [first mention, second mention]
                    # The dummy mention will be "epsilon"
                    mention_id_links = []

                    # Used for tracking mentions and their cluster, for constructing mention links
                    character2mention_dict = {}

                    for utterance in scene["utterances"]:
                        speaker_text = ",".join(utterance["speakers"])+":"

                        speaker_name = "_".join(utterance["speakers"])
                        speaker_id = speaker_name_to_ids[speaker_name]

                        speaker_token_ids = tokenizer.encode(speaker_text, add_special_tokens=True)
                        speaker_tokens = tokenizer.convert_ids_to_tokens(speaker_token_ids)

                        # Add the speaker information directly into context
                        current_segment = all_scene_tokens[current_segment_count]
                        current_mask_segment = all_scene_token_masks[current_segment_count]
                        current_speaker_name_segment = all_scene_speaker_names[current_segment_count]
                        current_speaker_id_segment = all_scene_speaker_ids[current_segment_count]

                        if len(current_segment) + len(speaker_tokens) > 512:
                            # Put to add padding to current segment and put speaker information to next segment
                            while len(current_segment) < 512:
                                current_segment.append("[PAD]")
                                current_mask_segment.append(0)
                                current_speaker_name_segment.append("")
                                current_speaker_id_segment.append(0)

                            all_scene_tokens.append([])
                            all_scene_token_masks.append([])
                            all_scene_speaker_ids.append([])
                            all_scene_speaker_names.append([])

                            current_segment_count += 1

                            current_segment = all_scene_tokens[current_segment_count]
                            current_mask_segment = all_scene_token_masks[current_segment_count]
                            current_speaker_name_segment = all_scene_speaker_names[current_segment_count]
                            current_speaker_id_segment = all_scene_speaker_ids[current_segment_count]

                            current_segment.extend(speaker_tokens)
                            current_mask_segment.extend([1 for _ in speaker_tokens])
                            current_speaker_name_segment.extend([speaker_name for _ in speaker_tokens])
                            current_speaker_id_segment.extend([speaker_id for _ in speaker_tokens])

                        else:
                            # Put to current segment
                            current_segment.extend(speaker_tokens)
                            current_mask_segment.extend([1 for _ in speaker_tokens])
                            current_speaker_name_segment.extend([speaker_name for _ in speaker_tokens])
                            current_speaker_id_segment.extend([speaker_id for _ in speaker_tokens])


                        tokenized_text = utterance["tokens"]
                        mention_count = 0

                        for sentence_count, entity_in_sentence in enumerate(utterance["character_entities"]):
                            sentence = tokenized_text[sentence_count]
                            sentence_old2new_mapping = []
                            sentence_tokens = []
                            prior_tokens = 1

                            for token in sentence:
                                sentence_old2new_mapping.append(prior_tokens)
                                word_piece_token = tokenizer.encode(token, add_special_tokens=False)

                                prior_tokens += len(word_piece_token)
                                sentence_tokens.extend(word_piece_token)
                            sentence_old2new_mapping.append(prior_tokens)
                            sentence_tokens = tokenizer.convert_ids_to_tokens(sentence_tokens)

                            sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]
                            sentence_old2new_mapping.append(prior_tokens + 1)
                            # add sentence to segments
                            if len(current_segment) + len(sentence_tokens) > 512:
                                # Put to add padding to current segment and put speaker information to next segment
                                while len(current_segment) < 512:
                                    current_segment.append("[PAD]")
                                    current_mask_segment.append(0)
                                    current_speaker_name_segment.append("")
                                    current_speaker_id_segment.append(0)

                                all_scene_tokens.append([])
                                all_scene_token_masks.append([])
                                all_scene_speaker_ids.append([])
                                all_scene_speaker_names.append([])

                                current_segment_count += 1
                                current_segment = all_scene_tokens[current_segment_count]
                                current_mask_segment = all_scene_token_masks[current_segment_count]
                                current_speaker_name_segment = all_scene_speaker_names[current_segment_count]
                                current_speaker_id_segment = all_scene_speaker_ids[current_segment_count]


                                current_segment.extend(sentence_tokens)
                                current_mask_segment.extend([1 for _ in sentence_tokens])
                                current_speaker_name_segment.extend([speaker_name for _ in sentence_tokens])
                                current_speaker_id_segment.extend([speaker_id for _ in sentence_tokens])
                            else:
                                # Put to current segment
                                current_segment.extend(sentence_tokens)
                                current_mask_segment.extend([1 for _ in sentence_tokens])
                                current_speaker_name_segment.extend([speaker_name for _ in sentence_tokens])
                                current_speaker_id_segment.extend([speaker_id for _ in sentence_tokens])

                            if len(entity_in_sentence) > 0:
                                for triplets in entity_in_sentence:
                                    # ignore multiple references
                                    if len(triplets) > 3:
                                        continue
                                    mention_id = utterance["utterance_id"] + "_m" + format(mention_count + 1, '05d')

                                    # Put the mention with position into mention
                                    prior_tokens_in_segment = len(current_segment) - len(sentence_tokens)
                                    new_start = prior_tokens_in_segment + sentence_old2new_mapping[triplets[0]]
                                    new_end = prior_tokens_in_segment + sentence_old2new_mapping[triplets[1]]
                                    entity_name = triplets[2]

                                    mentionid2positions[mention_id] = len(mention_id_with_positions)
                                    mention_id_with_positions.append([mention_id, current_segment_count, new_start, new_end])

                                    if entity_name in character_name_to_id:
                                        mention_character_id.append(character_name_to_id[entity_name])
                                        all_mention_ids.append(character_name_to_id[entity_name])
                                    else:
                                        mention_character_id.append(len(character_name_to_id))
                                        all_mention_ids.append(len(character_name_to_id))


                                    mention_character_name.append(entity_name)

                                    # check character2mention dict to construct links, or construct dummy mention id
                                    if entity_name in character2mention_dict:
                                        for prev_mention in character2mention_dict[entity_name]:
                                            mention_id_links.append([prev_mention, mention_id,  mentionid2positions[prev_mention], mentionid2positions[mention_id]])

                                        character2mention_dict[entity_name].append(mention_id)
                                    else:
                                        mention_id_links.append([dummy_mention_id, mention_id, -1, mentionid2positions[mention_id]])
                                        character2mention_dict[entity_name] = [mention_id]

                                    mention_count += 1
                                    all_mention_count += 1

                    while len(current_segment) < 512:
                        current_segment.append("[PAD]")
                        current_mask_segment.append(0)
                        current_speaker_name_segment.append("")
                        current_speaker_id_segment.append(0)



                    # print(len(all_scene_tokens))
                    # for j in all_scene_tokens:
                    #     print(len(j))
                    # print(len(mention_id_links))
                    # print(mention_id_with_positions)
                    # print(character2mention_dict))
                    scene_result = {
                        "scene_id": scene["scene_id"],
                        "all_scene_tokens": all_scene_tokens,
                        "token_masks": all_scene_token_masks,
                        "mentions": mention_id_with_positions,
                        "links": mention_id_links,
                        "speaker_id": all_scene_speaker_ids,
                        "speaker_name": all_scene_speaker_names,
                        "mention_character_id": mention_character_id,
                        "mention_character_name": mention_character_name
                    }
                    batch_sizes.append(len(all_scene_tokens))
                    number_of_mentions.append(len(mention_id_with_positions))
                    number_of_links.append(len(mention_id_links))
                    fout.write(json.dumps(scene_result)+"\n")
                    
        print("number of mention ids:", Counter(all_mention_ids))

        print("max batch_size", max(batch_sizes))
        print("max number of mentions", max(number_of_mentions))
        print("max number of links", max(number_of_links))
        print("max number of speakers", len(speaker_id_to_names))

        print("=============")
    #                 break
    #             break
    #         break
    # break