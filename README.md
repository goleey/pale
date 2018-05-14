# Inroduction
The source code "pale_stage2.py" implements the mapping function of PALE with tensorflow. The first stage that is learning representations of source and target networks is done by https://github.com/tangjianpku/LINE.git or https://github.com/thunlp/OpenNE.git here.


# Run
`python pale_stage2.py source_embedding target_embedding 1 train_file validation_file test_file mapped_source_embedding`

Here are 7 arguments you need to provide, **first 6** of which are input(file paths):  
source_embedding: the nodes' embeddings in the source file  
target_embedding: the nodes' embeddings in the target file  
linear or non linear:1 for linear and  0 for non linear  
train_file: observed anchor links to train the mapping function  
validation_file: observed anchor links to tune the hyper parameters  
test_file: test the performance(mrr and hit@1)  
**the last one** is the mapping embedding test node in the of source network:
mapped_source_embedding: the mapped embedding
# File format
##### soure_embedding:
trained by the https://github.com/tangjianpku/LINE.git or https://github.com/thunlp/OpenNE.git

##### target_embedding:
trained by the https://github.com/tangjianpku/LINE.git or https://github.com/thunlp/OpenNE.git

##### linear or non linear:
1 for linear and  0 for non linear

##### train_file(ids don't have to be the same):
1 1  
2 3  
3 2  

##### validation_file:
4 5  
5 6  

##### test_file: a node with a candidate list. Please make sure that the first node of the list is the groundtruth(corresponding anchor node).
7 7 8 9 10 11  
8 8 11 7 9 10  
##### mapped_source_embedding
7 emb1 emb2 emb3 ...  
8 emb1 emb2 emb3 ...  
