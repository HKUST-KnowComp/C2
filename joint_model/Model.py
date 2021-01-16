import torch
import numpy as np
from torch import nn
from transformers import *
import time
import json
import  matplotlib.pyplot  as plt
from matplotlib.pyplot import  Line2D
from tqdm import  tqdm
from apex import amp

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    # print(len(named_parameters))
    # print([n for n, p in named_parameters if (p.requires_grad) and ("bias" not in n)])
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            try:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
            except:
                print(n)

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

    # print(layers)
    # print(ave_grads)


def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])



class JointLearningModel(nn.Module):
    def __init__(self, pre_trained):
        super(JointLearningModel, self).__init__()

        if pre_trained[:8] == "SpanBERT":
            self.encoder = AutoModel.from_pretrained(pre_trained)
        else:
            self.encoder = BertModel.from_pretrained(pre_trained)

        hidden_size = self.encoder.config.hidden_size

        self.mention_pair_score1 = nn.Linear(hidden_size * 2, hidden_size)
        self.mention_pair_score2 = nn.Linear(hidden_size, hidden_size // 2)
        self.mention_pair_score3 = nn.Linear(hidden_size // 2, 1)


        # self.dummy_mention = torch.rand((1, hidden_size)).cuda()
        self.dummy_mention = nn.Embedding(1, hidden_size)

        self.mention_score1 = nn.Linear(hidden_size, hidden_size // 2)
        self.mention_score2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.mention_score3 = nn.Linear(hidden_size // 4, 1)

        self.speaker_embedding = nn.Embedding(400, hidden_size)

        self.character_linking1 = nn.Linear(hidden_size, hidden_size // 2)
        self.character_linking2 = nn.Linear(hidden_size // 2, 18)



    def forward(self, batch_of_segments, token_masks, mentions_seg, mention_start, mention_end, speaker_ids,
                link_first_mention_index=None, link_second_mention_index=None, character_label=None):
        sequence_output, _ = self.encoder(input_ids=batch_of_segments, attention_mask=token_masks)

        start_representation = sequence_output[mentions_seg, mention_start]
        end_representation = sequence_output[mentions_seg, mention_end]

        speaker_representation = self.speaker_embedding(speaker_ids) # [num_seq, max_length, hidden_size]
        mention_speaker_representation = speaker_representation[mentions_seg, mention_start]


        mention_representations = start_representation + end_representation + mention_speaker_representation

        # Linking loss first
        activate = nn.ReLU()
        mention_character_logits = self.character_linking2(activate(self.character_linking1(mention_representations)))




        # Higher order
        epsilon_representation = self.dummy_mention(torch.tensor(0).cuda())
        all_mention_representations = torch.cat([epsilon_representation.view((1,-1)), mention_representations], 0)

        num_mentions = all_mention_representations.shape[0]

        # mention_scores = - torch.ones((num_mentions, num_mentions)) * 10000

        # 这玩意用tensor的就可以搞定，多站memery少时间
        relu = nn.ReLU()

        tensor_mention_representation_first = torch.stack([all_mention_representations]*all_mention_representations.shape[0])
        tensor_mention_representation_second = tensor_mention_representation_first.permute(1, 0, 2)

        mention_pair_representation = torch.cat([tensor_mention_representation_first, tensor_mention_representation_second], dim=2)


        mention_pair_scores = self.mention_pair_score3(relu(self.mention_pair_score2(relu(self.mention_pair_score1(mention_pair_representation))))).view(num_mentions,-1)
        mention_scores_first =self.mention_score3( relu(self.mention_score2(relu( self.mention_score1(tensor_mention_representation_first))))).view(num_mentions,-1)
        mention_scores_second = self.mention_score3(relu(self.mention_score2(relu( self.mention_score1(tensor_mention_representation_second))))).view(num_mentions,-1)


        mention_softmax_mask = torch.triu(torch.ones((num_mentions, num_mentions)),diagonal=0) * -10000

        mention_scores = mention_pair_scores + mention_scores_first + mention_scores_second  + mention_softmax_mask.cuda()
        # print(mention_scores.requires_grad)
        # print(mention_scores)
        # print(mention_pair_scores.requires_grad)
        if link_first_mention_index is None:
            # The evaluation mode
            link_result = torch.argmax(mention_scores, dim=1)
            character_result = torch.argmax(mention_character_logits, dim=1)
            return link_result, character_result
        else:
            # The training mode


            # Directly minimize the negative log-likelihood
            softmax = torch.nn.Softmax(dim=1)
            softmax_mentions_score = softmax(mention_scores)

            onehot_link_first_mention = torch.zeros(mention_scores[link_second_mention_index].shape)
            num_links = mention_scores[link_second_mention_index].shape[0]
            onehot_link_first_mention[range(num_links), link_first_mention_index] = 1
            # print(onehot_link_first_mention)
            # print(onehot_link_first_mention.shape)

            likelihood = softmax_mentions_score[link_second_mention_index].cuda() * onehot_link_first_mention.cuda()    # [num_links, num_mention]

            sum_likelihood_of_each_link = likelihood.sum(1).view((1, -1)) # [1, num_links]
            # print(sum_likelihood_of_each_second_mention)

            # 每个第二个mention认领得到的分数并求和，用矩阵乘法实现.
            multihot_for_second_mention_index = torch.zeros(mention_scores[link_second_mention_index].shape) # [num_links, num_mentions]
            multihot_for_second_mention_index[range(num_links), link_second_mention_index] = 1

            sum_likelihood_of_each_second_mention = torch.matmul(sum_likelihood_of_each_link, multihot_for_second_mention_index.cuda())
            # print(sum_likelihood_of_each_second_mention.shape)
            log_likelihood = torch.log(sum_likelihood_of_each_second_mention[:,1:])
            sum_neg_log_likelihood = - log_likelihood.sum()


            # Linking loss
            linking_loss_fnc = nn.CrossEntropyLoss(reduction="sum")
            linking_loss = linking_loss_fnc(mention_character_logits, character_label)


            # print(sum_neg_log_likelihood)
            return sum_neg_log_likelihood + linking_loss

class JointLearningMemoryModel(nn.Module):
    def __init__(self, pre_trained, num_memory_layers, coref_weight=1, linking_weight=1):
        super(JointLearningMemoryModel, self).__init__()

        if pre_trained[:8] == "SpanBERT":
            self.encoder = AutoModel.from_pretrained(pre_trained)
        else:
            self.encoder = BertModel.from_pretrained(pre_trained)

        hidden_size = self.encoder.config.hidden_size

        self.mention_pair_score1 = nn.Linear(hidden_size * 2, hidden_size)
        self.mention_pair_score2 = nn.Linear(hidden_size, hidden_size // 2)
        self.mention_pair_score3 = nn.Linear(hidden_size // 2, 1)

        # self.dummy_mention = torch.rand((1, hidden_size)).cuda()
        self.dummy_mention = nn.Embedding(1, hidden_size)

        self.mention_score1 = nn.Linear(hidden_size, hidden_size // 2)
        self.mention_score2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.mention_score3 = nn.Linear(hidden_size // 4, 1)

        self.speaker_embedding = nn.Embedding(400, hidden_size)

        self.character_linking1 = nn.Linear(hidden_size, hidden_size // 2)
        self.character_linking2 = nn.Linear(hidden_size // 2, 18)

        # self.memory_capacity = memory_capacity
        # self.memory_keys = nn.Linear(hidden_size, memory_capacity,  bias=False)
        # self.memory_values = nn.Linear(memory_capacity, hidden_size, bias=False)
        #
        self.num_memory_layers = num_memory_layers
        # self.memory_H = nn.Linear(hidden_size, hidden_size, bias=False)

        self.mention_self_attention = BertLayer(self.encoder.config)

        self.coref_weight = coref_weight
        self.linking_weight = linking_weight


    def forward(self, batch_of_segments, token_masks, mentions_seg, mention_start, mention_end, speaker_ids,
                link_first_mention_index=None, link_second_mention_index=None, character_label=None):
        sequence_output, _ = self.encoder(input_ids=batch_of_segments, attention_mask=token_masks)

        start_representation = sequence_output[mentions_seg, mention_start]
        end_representation = sequence_output[mentions_seg, mention_end]

        speaker_representation = self.speaker_embedding(speaker_ids)  # [num_seq, max_length, hidden_size]
        mention_speaker_representation = speaker_representation[mentions_seg, mention_start]

        mention_representations = start_representation + end_representation + mention_speaker_representation

        # softmax = nn.Softmax(dim=1)
        unsqueezed_mention_representations = torch.unsqueeze(mention_representations, 0)
        for _ in range(self.num_memory_layers):
            layer_outputs = self.mention_self_attention(unsqueezed_mention_representations)
            unsqueezed_mention_representations = layer_outputs[0]
            # u = mention_representations
            # memory_soft_indices = softmax(self.memory_keys(mention_representations)) # [num_mentions, memory_capacity]
            # o = self.memory_values(memory_soft_indices)
            # mention_representations = self.memory_H(u) + o

        mention_representations = torch.squeeze(unsqueezed_mention_representations, 0)
        # Linking loss first
        activate = nn.ReLU()
        mention_character_logits = self.character_linking2(
            activate(self.character_linking1(mention_representations)))

        # Higher order
        epsilon_representation = self.dummy_mention(torch.tensor(0).cuda())
        all_mention_representations = torch.cat([epsilon_representation.view((1, -1)), mention_representations],
                                                0)

        num_mentions = all_mention_representations.shape[0]

        # mention_scores = - torch.ones((num_mentions, num_mentions)) * 10000

        # 这玩意用tensor的就可以搞定，多站memery少时间
        relu = nn.ReLU()

        tensor_mention_representation_first = torch.stack(
            [all_mention_representations] * all_mention_representations.shape[0])
        tensor_mention_representation_second = tensor_mention_representation_first.permute(1, 0, 2)

        mention_pair_representation = torch.cat(
            [tensor_mention_representation_first, tensor_mention_representation_second], dim=2)

        mention_pair_scores = self.mention_pair_score3(
            relu(self.mention_pair_score2(relu(self.mention_pair_score1(mention_pair_representation))))).view(
            num_mentions, -1)
        mention_scores_first = self.mention_score3(
            relu(self.mention_score2(relu(self.mention_score1(tensor_mention_representation_first))))).view(
            num_mentions, -1)
        mention_scores_second = self.mention_score3(
            relu(self.mention_score2(relu(self.mention_score1(tensor_mention_representation_second))))).view(
            num_mentions, -1)

        mention_softmax_mask = torch.triu(torch.ones((num_mentions, num_mentions)), diagonal=0) * -10000

        mention_scores = mention_pair_scores + mention_scores_first + mention_scores_second + mention_softmax_mask.cuda()
        # print(mention_scores.requires_grad)
        # print(mention_scores)
        # print(mention_pair_scores.requires_grad)
        if link_first_mention_index is None:
            # The evaluation mode
            link_result = torch.argmax(mention_scores, dim=1)
            character_result = torch.argmax(mention_character_logits, dim=1)
            return link_result, character_result
        else:
            # The training mode


            # Directly minimize the negative log-likelihood
            softmax = torch.nn.Softmax(dim=1)
            softmax_mentions_score = softmax(mention_scores)

            onehot_link_first_mention = torch.zeros(mention_scores[link_second_mention_index].shape)
            num_links = mention_scores[link_second_mention_index].shape[0]
            onehot_link_first_mention[range(num_links), link_first_mention_index] = 1
            # print(onehot_link_first_mention)
            # print(onehot_link_first_mention.shape)

            likelihood = softmax_mentions_score[
                             link_second_mention_index].cuda() * onehot_link_first_mention.cuda()  # [num_links, num_mention]

            sum_likelihood_of_each_link = likelihood.sum(1).view((1, -1))  # [1, num_links]
            # print(sum_likelihood_of_each_second_mention)

            # 每个第二个mention认领得到的分数并求和，用矩阵乘法实现.
            multihot_for_second_mention_index = torch.zeros(
                mention_scores[link_second_mention_index].shape)  # [num_links, num_mentions]
            multihot_for_second_mention_index[range(num_links), link_second_mention_index] = 1

            sum_likelihood_of_each_second_mention = torch.matmul(sum_likelihood_of_each_link,
                                                                 multihot_for_second_mention_index.cuda())
            # print(sum_likelihood_of_each_second_mention.shape)
            log_likelihood = torch.log(sum_likelihood_of_each_second_mention[:, 1:])
            sum_neg_log_likelihood = - log_likelihood.sum()

            # Linking loss
            linking_loss_fnc = nn.CrossEntropyLoss(reduction="sum")
            linking_loss = linking_loss_fnc(mention_character_logits, character_label)

            # print(sum_neg_log_likelihood)
            return self.coref_weight * sum_neg_log_likelihood +  self.linking_weight * linking_loss


# The training and testing are both conducted in a document-as-batch manner.
class dataloader():
    def __init__(self, data_path, tokenizer):
        self.mention_ids = []
        with open(data_path, "r") as fin:
            self.all_documents = []
            for line in tqdm(fin):
                document = json.loads(line.strip())

                batch_ids = [tokenizer.convert_tokens_to_ids(seg) for seg in document["all_scene_tokens"]]

                speaker_ids = document["speaker_id"]
                character_ids = document["mention_character_id"]

                mention_ids = [m[0] for m in document["mentions"]]
                mention_seg = [m[1] for m in document["mentions"]]
                mention_start = [m[2] for m in document["mentions"]]
                mention_end = [m[3]-1 for m in document["mentions"]]

                masks = document["token_masks"]

                link_start = [l[2] + 1 for l in document["links"]]
                link_end = [l[3] + 1 for l in document["links"]]

                batch_ids = torch.tensor(batch_ids)
                masks = torch.tensor(masks)
                mention_seg = torch.tensor(mention_seg)
                mention_start = torch.tensor(mention_start)
                mention_end = torch.tensor(mention_end)

                link_start = torch.tensor(link_start)
                link_end = torch.tensor(link_end)

                speaker_ids = torch.tensor(speaker_ids)

                character_ids = torch.tensor(character_ids)

                self.all_documents.append([batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids])
                self.mention_ids.append(mention_ids)
        self.length = len(self.all_documents)

        self.current_id = 0

    def get_document(self):
        result = self.all_documents[self.current_id]
        resulting_mentions = self.mention_ids[self.current_id]
        self.current_id += 1
        if self.current_id >= self.length:
            self.current_id = 0

        # Directly to Cuda
        return [x.cuda() for x in result], resulting_mentions

    def reset(self):
        self.current_id = 0


if __name__ == "__main__":
    model = JointLearningModel("SpanBERT/spanbert-large-cased")
    tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")

    n_gpu = torch.cuda.device_count()

    model.cuda()

    optimizer = AdamW(model.parameters(), lr=0.0005, correct_bias=False)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

  
    documents = []
    with open("./data/trn_SpanBERTspanbert-large-cased_batches.json") as fin:
        for line in fin:
            document = json.loads(line.strip())
            documents.append(document)

    # If the card can put these two documents in to the memory, then it would work for all documents

    longest_document = [d for d in documents if len(d["all_scene_tokens"]) == 8][0]
    # print(len(longest_document["mentions"]))
    # print(len(longest_document["links"]))

    most_links_document = [d for d in documents if len(d["links"]) == 7644][0]
    most_links_document = documents[0]
    # print(len(most_links_document["all_scene_tokens"]))
    # print(len(most_links_document["links"]))

    batch_ids = [tokenizer.convert_tokens_to_ids(seg) for seg in longest_document["all_scene_tokens"]]

    masks = longest_document["token_masks"]

    # print([len(l) for l in masks])

    mention_ids = [m[0] for m in  longest_document["mentions"]]
    mention_seg = [m[1] for m in  longest_document["mentions"]]
    mention_start = [m[2] for m in longest_document["mentions"]]
    mention_end = [m[3] for m in longest_document["mentions"]]

    # All add + 1 because there is a dummy mention
    link_start = [l[2]+1 for l in longest_document["links"]]
    link_end = [l[3]+1 for l in longest_document["links"]]

    speaker_ids = [s for s in longest_document["speaker_id"]]

    character_ids = [s for s in longest_document["mention_character_id"]]

    # print([len(l) for l in speaker_ids])

    batch_ids = torch.tensor(batch_ids).cuda()
    masks = torch.tensor(masks).cuda()
    mention_seg = torch.tensor(mention_seg).cuda()
    mention_start = torch.tensor(mention_start).cuda()
    mention_end = torch.tensor(mention_end).cuda()

    speaker_ids = torch.tensor(speaker_ids).cuda()

    link_start = torch.tensor(link_start).cuda()
    link_end = torch.tensor(link_end).cuda()

    character_ids = torch.tensor(character_ids).cuda()



    output = model(batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids)
    #
    # with amp.scale_loss(output, optimizer) as scaled_loss:
    #     scaled_loss.backward()

    output.backward()
    # trn_data = "/home/yons/CharacterRelationMining/baselines/c2f/data/dev_SpanBERTspanbert-base-cased_batches.json"
    #
    # train_dataloader = dataloader(trn_data, tokenizer)
    #
    optimizer.step()
    optimizer.zero_grad()

    # print(train_dataloader.get_document())


    time.sleep(10)

