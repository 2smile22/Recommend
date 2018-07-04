# Deep Learning and Recommenders
## 2018 papers
- [Dongmin Hyun, Chanyoung Park, Min-Chul Yang, Ilhyeon Song, Jung-Tae Lee and Hwanjo Yu.The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '18 short paper) (To appear)](http://dm.postech.ac.kr/~pcy1302/data/SIGIR18.pdf) _SIGIR_
- 推荐系统可以根据已有评分和其他信息来预测给定用户对给定商品的评分。最成功的方法是协同过滤，但是会面临数据稀疏和冷启动的问题。最近，深度学习模型被提出用来缓解使用评论的评分矩阵的稀疏性，然而由于文本中存在歧义，已有的一些方法仅仅根据评论中生词的串联是具有挑战性的。因此，情感信息对于建模用户和商品来解决评论中的不确定性至关重要。由于对评论的过度表示，它们在训练时间和内存使用方面也不具有可扩展性。本文提出了一种可伸缩的评论推荐方法，称为SentiRec，在建模用户和商品时结合评论的情感信息。SentiRec方法有两步组成：1.纳入评论情感，包括将每个评论编码为体现评论情感的固定长度评论向量；2.根据向量编码的评论生成推荐。
## 2017 papers
- [Li X, She J. Collaborative Variational Autoencoder for Recommender Systems[C]// The, ACM SIGKDD International Conference. ACM, 2017:305-314.](https://dl.acm.org/citation.cfm?doid=3097983.3098077]) _SIGKDD_
- 现有的推荐系统方法大致可分为三类:基于内容的方法、协同过滤方法和混合方法。协同过滤方法具有稀疏性、冷启动等缺点，混合方法考虑了评分和内容信息。越来越多的人开始关注混合方法，它们同时考虑评分和内容信息，以往的研究大多不能学习推荐任务内容的良好表示，或者只考虑内容的文本模式，因此这些方法在当前的多媒体场景（如文本、图像）中是非常有限的。本文提出了一种贝叶斯生成模型称为collaborative variational autoencoder(CVAE)，该模型考虑了多媒体场景中推荐的评分和内容。模型以一种无监督的方式从内容数据中学习深层次的潜在表达形式，并从内容和评分两个方面学习商品和用户之间的隐式关系。与以往的去噪标准不同的是，所提出的CVAE通过推理网络获取了隐藏空间（非观测空间）中内容的潜在分布，可以很容易地扩展到除文本之外的其他多媒体模式。
- [Li J, Ren P, Chen Z, et al. Neural Attentive Session-based Recommendation[J]. 2017.CIKM’17 , November 6–10, 2017, Singapore.](https://dl.acm.org/citation.cfm?id=3132926) _CIKM_
- 关键词：Session-based recommendation, sequential behavior, recurrent neural networks, attention mechanism
- 传统的两类推荐方法如基于内容的推荐算法和协同过滤推荐算法，它们在刻画序列数据中存在缺陷：每个item相互独立，不能建模session中item的连续偏好信息。传统方法采用session中item间的相似性预测下一个item，它的缺点是只考虑了最后一次的点击的item相似性，忽视了前面的点击, 没有考虑整个序列信息。
本文文考虑到用户当前会话的行为序列和主要意图，提出了一个新的NARM模型。首次将RNN运用于Session-based(将其理解为具有时序关系的一些记录序列)推荐，一个session 中点击 item 的行为看做一个序列，用GRU来刻画。针对该任务设计了RNN的训练、评估方法及ranking loss。
## 2016 papers
- 
