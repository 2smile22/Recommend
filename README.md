# Deep Learning and Recommenders
## 2018 papers
- [Dongmin Hyun, Chanyoung Park, Min-Chul Yang, Ilhyeon Song, Jung-Tae Lee and Hwanjo Yu.The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '18 short paper) (To appear)](http://dm.postech.ac.kr/~pcy1302/data/SIGIR18.pdf) _SIGIR_
- 推荐系统可以根据已有评分和其他信息来预测给定用户对给定商品的评分。最成功的方法是协同过滤，但是会面临数据稀疏和冷启动的问题。最近，深度学习模型被提出用来缓解使用评论的评分矩阵的稀疏性，然而由于文本中存在歧义，已有的一些方法仅仅根据评论中生词的串联是具有挑战性的。因此，情感信息对于建模用户和商品来解决评论中的不确定性至关重要。由于对评论的过度表示，它们在训练时间和内存使用方面也不具有可扩展性。本文提出了一种可伸缩的评论推荐方法，称为SentiRec，在建模用户和商品时结合评论的情感信息。SentiRec方法有两步组成：1.纳入评论情感，包括将每个评论编码为体现评论情感的固定长度评论向量；2.根据向量编码的评论生成推荐。
- [Tang J, Wang K. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding[J]. 2018.](http://www.sfu.ca/~jiaxit/resources/wsdm18caser.pdf) _WSDM_
- 关键词：Recommender System; Sequential Prediction; Convolutional Neural Networks
- 推荐系统已经成为许多应用的核心技术。大多数系统，如top-N推荐，都会根据用户的一般偏好来推荐商品，而不关注商品的新近程度。Top-N顺序推荐对每个用户过去互动的商品序列进行建模，并旨在预测用户在“不远的将来”可能会进行交互的排名Top-N的商品。交互的顺序意味着，当序列中最近的商品对下一个商品有更大的影响时，序列模式起着重要的作用。本文提出了一个卷积序列嵌入推荐模型(Caser)作为解决这一需求的解决方案。其想法是将最近的商品序列嵌入到时间和潜在空间中的“图像”中，并通过卷积过滤器学习序列模式作为图像的局部特征。这种方法提供了统一和灵活的网络结构来去捕获一般偏好和序列模式。
## 2017 papers
- [Li X, She J. Collaborative Variational Autoencoder for Recommender Systems[C]// The, ACM SIGKDD International Conference. ACM, 2017:305-314.](https://dl.acm.org/citation.cfm?doid=3097983.3098077]) _SIGKDD_
- 现有的推荐系统方法大致可分为三类:基于内容的方法、协同过滤方法和混合方法。协同过滤方法具有稀疏性、冷启动等缺点，混合方法考虑了评分和内容信息。越来越多的人开始关注混合方法，它们同时考虑评分和内容信息，以往的研究大多不能学习推荐任务内容的良好表示，或者只考虑内容的文本模式，因此这些方法在当前的多媒体场景（如文本、图像）中是非常有限的。本文提出了一种贝叶斯生成模型称为collaborative variational autoencoder(CVAE)，该模型考虑了多媒体场景中推荐的评分和内容。模型以一种无监督的方式从内容数据中学习深层次的潜在表达形式，并从内容和评分两个方面学习商品和用户之间的隐式关系。与以往的去噪标准不同的是，所提出的CVAE通过推理网络获取了隐藏空间（非观测空间）中内容的潜在分布，可以很容易地扩展到除文本之外的其他多媒体模式。
- [Li J, Ren P, Chen Z, et al. Neural Attentive Session-based Recommendation[J]. 2017.CIKM’17 , November 6–10, 2017, Singapore.](https://dl.acm.org/citation.cfm?id=3132926) _CIKM_
- 关键词：Session-based recommendation, sequential behavior, recurrent neural networks, attention mechanism
- 传统的两类推荐方法如基于内容的推荐算法和协同过滤推荐算法，它们在刻画序列数据中存在缺陷：每个item相互独立，不能建模session中item的连续偏好信息。传统方法采用session中item间的相似性预测下一个item，它的缺点是只考虑了最后一次的点击的item相似性，忽视了前面的点击, 没有考虑整个序列信息。
本文文考虑到用户当前会话的行为序列和主要意图，提出了一个新的NARM模型。首次将RNN运用于Session-based(将其理解为具有时序关系的一些记录序列)推荐，一个session 中点击 item 的行为看做一个序列，用GRU来刻画。针对该任务设计了RNN的训练、评估方法及ranking loss。
- [Li Y, Chen W, Yan H. Learning Graph-based Embedding For Time-Aware Product Recommendation[C]// ACM, 2017:2163-2166. CIKM’17, November 6-10, 2017, Singapore](https://dl.acm.org/citation.cfm?doid=3132847.3133060) _CIKM_
- 关键词：Network Embedding; Product Recommendation; Dynamic User Embedding; Time Aware
- 目前，各种推荐系统的方法被提出，最初是基于人口统计学，内容和协同过滤。尽管各种方法在一些领域具有高效性，但是它们通常在较小的数据集上表现良好，当数据的规模不断增加时，问题就变得很有挑战性了。针对大规模数据集，几种基于网络嵌入的推荐算法已经被提出，然而它们不能高效的表示用户的动态偏好。
本文提出了一种新的产品图嵌入模型(PGE)，利用网络表示学习技术来研究时间感知的产品推荐。本文的模型通过将历史购买记录转换为产品图来捕获产品的顺序影响。然后利用网络嵌入模型将产品转化为低维向量。一旦产品被映射到潜在空间中，本文提出了一种计算用户最新偏好的新方法，它将用户投射到与产品相同的潜在空间中。一旦获得产品嵌入，采用时间衰减函数来跟踪用户偏好的动态。最终的产品推荐是基于产品的嵌入以及在相同的潜在空间中的用户偏好。
## 2016 papers
- [Song Y, Elkahky A M, He X. Multi-rate deep learning for temporal recommendation[C]//Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2016: 909-912.](http://delivery.acm.org/10.1145/2920000/2914726/p909-song.pdf?ip=202.113.176.146&id=2914726&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EE4E04C281054793F%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1530646781_035154603033357daaa028281db1643d) _SIGIR_
- 在推荐系统中建模时间行为是一个重要而具有挑战性的问题。它的挑战来自于时间建模增加了参数估计和推理的成本，同时需要大量的数据来可靠地学习模型和额外的时间维度。因此，在大型的现实世界推荐系统中，通常很难对时间行为进行建模。
本文提出了一种新的基于时间深度神经网络的体系结构，该体系结构将长期静态和短期用户偏好相结合，以提高推荐性能。为了对大型应用程序进行有效的训练，本文还提出了一种新颖的预训练方法，可以显著减少自由参数的数量。
