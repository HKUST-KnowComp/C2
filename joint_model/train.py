from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import time
import datetime
from pathlib import Path

from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import BertForPreTraining, BertConfig, BertForMaskedLM

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from Model import JointLearningMemoryModel, dataloader
import os
import sys
sys.path.append(os.path.abspath("/home/yons/CharacterRelationMining/coref_model"))

from evaluators import Ceaf4_score, Bcube_score, MUC_score, links2clusters_new, score2clusters_new, Blanc_score, links2clusters, links2clusters, score2clusters
from transformers import *


import datetime
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file_name = './logs/log-'+now_time
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    filename = log_file_name,
                    filemode = 'w',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def softmax_result_to_cluster(prev_links, mentions, prev_ids):
    # prev_ids include the dummy mentions, so need to
    # print(len(prev_ids))
    # print(len(mentions))
    for position, second_mention in enumerate(mentions):
        if prev_ids[position] == -1:
            continue
        first_mention = mentions[prev_ids[position]]
        prev_links[(first_mention, second_mention)] = 1


    return prev_links



def main():
    parser = ArgumentParser()

    parser.add_argument('--trn_data', type=Path, required=True)
    parser.add_argument('--dev_data', type=Path, required=True)
    parser.add_argument('--tst_data', type=Path, required=True)

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--model",
                        required=True,
                        type=str,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_epochs",
                        default=10,
                        type=int,
                        help="Total number of epochs for training.")

    parser.add_argument("--num_memory_layers",
                        default=2,
                        type=int,
                        help="Total number of layers in memory network .")

    parser.add_argument("--coref_weight",
                        default=1,
                        type=float,
                        help="The weight in the front of coref loss.")

    parser.add_argument("--linking_weight",
                        default=1,
                        type=float,
                        help="The weight in the front of linking loss.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="# Gradient clipping is not in AdamW anymore (so you can use amp without issue)")

    parser.add_argument('--output_examples',
                        action='store_true',
                        help="output some examples during training to do case study")

    parser.add_argument('--dev_keys', type=Path, required=True)
    parser.add_argument('--trn_keys', type=Path, required=True)
    parser.add_argument('--tst_keys', type=Path, required=True)

    args = parser.parse_args()
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {},  16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model = JointLearningMemoryModel(args.model, args.num_memory_layers, args.coref_weight, args.linking_weight)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")



    if args.model[:8] == "SpanBERT":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model)

    train_dataloader = dataloader(args.trn_data, tokenizer)
    dev_dataloader = dataloader(args.dev_data, tokenizer)
    test_dataloader = dataloader(args.tst_data, tokenizer)

    num_training_steps = train_dataloader.length * args.num_epochs
    num_warmup_steps = args.warmup_proportion * num_training_steps

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)



    with open(args.trn_keys, "r") as fin:
        train_triplet = json.load(fin)
        gold_train_cluster = links2clusters(train_triplet)

    with open(args.dev_keys, "r") as fin:
        dev_triplet = json.load(fin)
        dev_triplet_dict = {(trip[0], trip[1]): trip[2] for trip in dev_triplet}
        gold_dev_cluster = links2clusters(dev_triplet)

    with open(args.tst_keys, "r") as fin:
        test_triplet = json.load(fin)
        test_triplet_dict = {(trip[0], trip[1]): trip[2] for trip in test_triplet}
        gold_test_cluster = links2clusters(test_triplet)


    for epoch_counter in range(args.num_epochs):
        train_dataloader.reset()
        dev_dataloader.reset()
        test_dataloader.reset()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        logger.info("***** Running training *****")
        logger.info("Epoch: " + str(epoch_counter))
        # First training
        model.train()
        with tqdm(total=train_dataloader.length, desc=f"Trn Epoch {epoch_counter}") as pbar:
            for _ in range(train_dataloader.length):
                input_values, mention_ids = train_dataloader.get_document()
                batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids = input_values
                if mention_start.shape[0] > 0:
                    try:
                        loss = model(batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids)

                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        tr_loss += loss.item()
                        nb_tr_examples += mention_start.size(0)
                        nb_tr_steps += 1

                        mean_loss = tr_loss / nb_tr_examples

                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()

                        optimizer.zero_grad()
                    except:
                        print("cannot fit in:", _)
                        pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                        pbar.update(1)
                        continue


                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                pbar.update(1)
        del loss


        model.eval()

        if epoch_counter % 10 == 0 :
            logger.info("***** Running evaluation *****")
            with tqdm(total=test_dataloader.length, desc=f"Trn Epoch {epoch_counter}") as pbar2:

                all_prev_link_dict = {t: 0 for t in test_triplet_dict.keys()}
                all_outputs = []
                all_labels = []


                for _ in range(test_dataloader.length):
                    input_values, mention_ids = test_dataloader.get_document()
                    batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids = input_values

                    if mention_start.shape[0] > 0:
                        link_result, linking_result = model(batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids)

                        prev_positions = [i - 1 for i in link_result.cpu().detach().numpy()]
                        prev_positions = prev_positions[1:]

                        all_outputs.extend(linking_result.cpu().detach().numpy().tolist())
                        all_labels.extend(character_ids.cpu().numpy().tolist())

                        # document_clusters = softmax_result_to_cluster(mention_ids, prev_positions)
                        all_prev_link_dict =  softmax_result_to_cluster (all_prev_link_dict, mention_ids, prev_positions)

                    pbar2.update(1)

                triplet_scores = [[t[0], t[1], score] for t, score in all_prev_link_dict.items()]

                all_clusters = links2clusters(triplet_scores)

                logger.info("Linking accuracy: " + str(accuracy_score(all_labels, all_outputs)))
                logger.info("Linking macro Pr: " + str(precision_score(all_labels, all_outputs, average='macro')))
                logger.info("Linking micro Pr: " + str(precision_score(all_labels, all_outputs, average="micro")))
                logger.info("Linking macro Rc: " + str(recall_score(all_labels, all_outputs, average='macro')))
                logger.info("Linking micro Rc: " + str(recall_score(all_labels, all_outputs, average="micro")))
                logger.info("Linking macro F1: " + str(f1_score(all_labels, all_outputs, average='macro')))
                logger.info("Linking micro F1: " + str(f1_score(all_labels, all_outputs, average="micro")))

                logger.info("Linking class F1: " + str(f1_score(all_labels, all_outputs, average=None)))

                # logger.info("MUC___score: " + str(MUC_score(gold_test_cluster, all_clusters)))
                logger.info("Bcube_score: " + str(Bcube_score(gold_test_cluster, all_clusters)))
                logger.info("Ceaf4_score: " + str(Ceaf4_score(gold_test_cluster, all_clusters)))
                logger.info("Blanc_score: " + str(Blanc_score(gold_test_cluster, all_clusters)))
                logger.info(f"Loss: {mean_loss:.5f}")
            del link_result, linking_result

            if args.output_examples:
                with open("./logs/dev_samples.json", "w") as fout:
                    logger.info("***** Saving Some Dev Results *****")
                    with tqdm(total=dev_dataloader.length, desc=f"Dev Epoch {epoch_counter}") as pbar3:

                        for _ in range(dev_dataloader.length):
                            input_values, mention_ids = dev_dataloader.get_document()
                            batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids = input_values

                            if mention_start.shape[0] > 0:
                                link_result, linking_result = model(batch_ids, masks, mention_seg, mention_start, mention_end,
                                                                    speaker_ids)

                                coref_prev_positions = [int(i - 1) for i in link_result.cpu().detach().numpy()]
                                coref_prev_positions = coref_prev_positions[1:]

                                linking_output_character_id = linking_result.cpu().detach().numpy().tolist()
                                linking_output_character_label = character_ids.cpu().numpy().tolist()

                                linking_output_character_id = [int(i) for i in linking_output_character_id]
                                linking_output_character_label = [int(i) for i in linking_output_character_label]

                                dev_result = {
                                    "mention_ids": mention_ids,
                                    "coref_prev_positions":coref_prev_positions,
                                    "linking_output_character_id": linking_output_character_id
                                }

                                fout.write(json.dumps(dev_result) + "\n")
                            else:
                                dev_result = {
                                    "mention_ids": [],
                                    "coref_prev_positions": [],
                                    "linking_output_character_id": []
                                }
                                fout.write(json.dumps(dev_result) + "\n")


                            pbar3.update(1)



                        del link_result, linking_result







if __name__ == "__main__":
    main()