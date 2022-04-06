프로젝트 설명 블로그 : https://www.notion.so/USE-WEAK-TOXIC-6e2254aed4884e909800b1b89fffa2c4

Dataset 설명

- tok : tokenization
- aug : Augmentation( Use GoogleTrans)
- con : only context
- bi/tri : binary / Multinomial(Tri)
- ex) merged_bi.train.tok.tsv -> Tokenized Binary Classification Dataset for Train

Models 설명

- bi : Binary Classification Model
- tri : Multinomial Classification Model
- tri_aug : Multinomial Classification Model with Augmented Dataset
- bi_con : Binarry Classification use only context-response data
