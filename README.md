# C2
The implementation for the paper Joint Coreference Resolution and Character Linking for Multiparty Conversation 
https://arxiv.org/abs/2101.11204

## Preprocess the character-identification data
`python preprocess_tokenization.py` \
`python preprocess_mention_cluster.py` \
`python preprocess_joint.py`


## Train the model
To train the base model with fp16, you can run \
`bash train_spanbert_base.sh` \
If you did not have install apex, you can remove the flag of `--fp16`.

To run spanbert_large model you need to have at least 24GB memory GPU\
`bash train_spanbert_large.sh` 


