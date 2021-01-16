# C2
The implementation for the paper: Joint Coreference Resolution and Character Linkingfor Multiparty Conversation


## Preprocess the character-identification data
`python preprocess_tokenization.py` \
`python preprocess_mention_cluster.py` \
`python preprocess_joint.py`


## Train the model
To train the base model with fp16, you can run.
If you did not have install apex, you can remove the flag of `--fp16`.
`bash train_spanbert_base.sh` \

To run spanbert_large model you need to have at least 24GB memory GPU\
`bash train_spanbert_large.sh` 


