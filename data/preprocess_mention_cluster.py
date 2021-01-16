# This file is to process the data file to make the

import json
from tqdm import tqdm
import numpy as np


data_path = ["trn_bert_base_cased.json", "dev_bert_base_cased.json", "tst_bert_base_cased.json"]
data_name = ["trn", "dev", "tst"]

for i in range(3):
    data_file_path = data_path[i]

    with open(data_file_path, "r") as fin:
        data = json.load(fin)

    mentions = []
    utterance_lengths = []
    sentence_lengths = []

    # The dictionary that record {mention_id: context, speaker, mention position, identities}
    mention_dict = {}
    cluster_scene = {}
    cluster_episode = {}

    cluster_scene_count = 0
    cluster_episode_count = 0

    data_set_keys_scene = []
    data_set_keys_episode = []

    scene_line_count = 0
    episode_line_count = 0

    for epi in tqdm(data["episodes"]):
        episode_character_mention_dict = {}
        episode_mention_character_dict = {}

        # Keep track of previous utterance and speakers
        prev_utterance_tokens_1 = []
        prev_utterance_tokens_2 = []

        prev_speaker_1 = []
        prev_speaker_2 = []
        # sequence: prev_utterance_1, prev_utterance_2, curr_utterance

        for scene in epi["scenes"]:
            prev_utterance_tokens_1 = []
            prev_utterance_tokens_2 = []

            prev_speaker_1 = []
            prev_speaker_2 = []

            scene_character_mention_dict = {}
            scene_mention_character_dict = {}

            # Used to add the next utterance information to a previous mention context
            # Because the context / speaker of the next utterance is also very important
            mention_ids_prev_utterance = []

            for utterance in scene["utterances"]:
                for prev_mention_id in mention_ids_prev_utterance:
                    mention_dict[prev_mention_id]["next_utterance_tokens"] = \
                        sum(utterance["tokens_bert_base_cased"], [])
                    mention_dict[prev_mention_id]["next_speaker"] = utterance["speakers"]

                mention_ids_prev_utterance = []

                mention_count = 0
                for sentence_count, entity_in_sentence in enumerate(utterance["character_entities_bert_base_cased"]):
                    if len(entity_in_sentence) > 0:
                        for triplets in entity_in_sentence:
                            # ignore multiple references
                            if len(triplets) > 3:
                                continue
                            mention_id = utterance["utterance_id"] + "_m" + format(mention_count+1, '05d')
                            mention_ids_prev_utterance.append(mention_id)

                            # The number of tokens in the previous sentences in the utterance.
                            # The index writen in the triplet is only the order in the current sentence
                            prev_tokens_count = 0
                            for sentence_index in range(0, sentence_count):
                                prev_tokens_count += len(utterance["tokens_bert_base_cased"][sentence_index])

                            mention_position = [triplets[0] + prev_tokens_count, triplets[1] + prev_tokens_count]
                            mention_identity = triplets[2]

                            utterance_tokens_bert_base_cased = sum(utterance["tokens_bert_base_cased"], [])

                            speakers = utterance["speakers"]

                            mention_dict[mention_id] = {
                                "mention_position": mention_position,
                                "mention_identity": mention_identity,
                                "utterance_tokens_bert_base_cased": utterance_tokens_bert_base_cased,
                                "speakers": speakers,
                                "prev_utterance_tokens_1": prev_utterance_tokens_1,
                                "prev_utterance_tokens_2": prev_utterance_tokens_2,
                                "prev_speaker_1": prev_speaker_1,
                                "prev_speaker_2": prev_speaker_2,
                                "next_utterance_tokens": [],
                                "next_speaker": []
                            }

                            if mention_identity in episode_character_mention_dict:
                                episode_character_mention_dict[mention_identity].append(mention_id)
                            else:
                                episode_character_mention_dict[mention_identity] = [mention_id]
                                cluster_episode_count += 1

                            if mention_identity in scene_character_mention_dict:
                                scene_character_mention_dict[mention_identity].append(mention_id)
                            else:
                                scene_character_mention_dict[mention_identity] = [mention_id]
                                cluster_scene_count += 1

                            for key, value in scene_mention_character_dict.items():
                                if value == mention_identity:
                                    data_set_keys_scene.append([key, mention_id, 1])
                                    scene_line_count += 1
                                else:
                                    data_set_keys_scene.append([key, mention_id, 0])

                            for key, value in episode_mention_character_dict.items():
                                if value == mention_identity:
                                    data_set_keys_episode.append([key, mention_id, 1])
                                    episode_line_count += 1
                                else:
                                    data_set_keys_episode.append([key, mention_id, 0])

                            # Update tracking variables
                            mention_count += 1
                            scene_mention_character_dict[mention_id] = mention_identity
                            episode_mention_character_dict[mention_id] = mention_identity

                prev_utterance_tokens_1 = prev_utterance_tokens_2
                prev_speaker_1 = prev_speaker_2
                prev_utterance_tokens_2 = sum(utterance["tokens_bert_base_cased"], [])
                prev_speaker_2 = utterance["speakers"]


            cluster_scene[scene["scene_id"]] = scene_character_mention_dict
        cluster_episode[epi["episode_id"]] = episode_character_mention_dict

    with open("mention_"+data_name[i]+".json", "w") as fout:
        json.dump(mention_dict, fout)

    with open("cluster_scene_"+data_name[i]+".json", "w") as fout:
        json.dump(cluster_scene, fout)
        print("Scene Clusters " + data_name[i], cluster_scene_count)

    with open("cluster_episode_"+data_name[i]+".json", "w") as fout:
        json.dump(cluster_episode, fout)
        print("Episode Clusters " + data_name[i], cluster_episode_count)

    with open("data_set_keys_scene_"+data_name[i]+".json", "w") as fout:
        json.dump(data_set_keys_scene, fout)
        print("Scene Level Samples "+data_name[i], scene_line_count, "/", len(data_set_keys_scene))

    with open("data_set_keys_episode_"+data_name[i]+".json", "w") as fout:
        json.dump(data_set_keys_episode, fout)
        print("Episode Level Samples " + data_name[i], episode_line_count, "/",  len(data_set_keys_episode))