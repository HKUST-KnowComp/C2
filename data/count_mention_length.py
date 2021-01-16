import json

names = ["mention_trn.json", "mention_dev.json", "mention_tst.json"]
for name in names:
    with open(name, "r") as fin:
        mention_dict = json.load(fin)

    prev1_lengths = [len(value["prev_utterance_tokens_1"]) for key, value in mention_dict.items()]
    prev2_lengths = [len(value["prev_utterance_tokens_2"]) for key, value in mention_dict.items()]
    curr_lengths = [len(value["utterance_tokens_bert_base_cased"]) for key, value in mention_dict.items()]
    next_lengths = [len(value["next_utterance_tokens"]) for key, value in mention_dict.items()]

    print(max(prev1_lengths))
    print(max(prev2_lengths))
    print(max(curr_lengths))
    print(max(next_lengths))

    print(len([i for i in curr_lengths if i > 250]), len(curr_lengths))