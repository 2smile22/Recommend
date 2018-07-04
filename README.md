# Deep Learning and Recommenders
## 2018 papers
- [Dongmin Hyun, Chanyoung Park, Min-Chul Yang, Ilhyeon Song, Jung-Tae Lee and Hwanjo Yu.The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '18 short paper) (To appear)](http://dm.postech.ac.kr/~pcy1302/data/SIGIR18.pdf) _SIGIR_
- 推荐系统可以根据已有评分和其他信息来预测给定用户对给定商品的评分。最成功的方法是协同过滤，但是会面临数据稀疏和冷启动的问题。最近，深度学习模型被提出用来缓解使用评论的评分矩阵的稀疏性，然而由于文本中存在歧义，已有的一些方法仅仅根据评论中生词的串联是具有挑战性的。因此，情感信息对于建模用户和商品来解决评论中的不确定性至关重要。由于对评论的过度表示，它们在训练时间和内存使用方面也不具有可扩展性。本文提出了一种可伸缩的评论推荐方法，称为SentiRec，在建模用户和商品时结合评论的情感信息。SentiRec方法有两步组成：1.纳入评论情感，包括将每个评论编码为体现评论情感的固定长度评论向量；2.根据向量编码的评论生成推荐。
- [Tang J, Wang K. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding[J]. 2018.](http://www.sfu.ca/~jiaxit/resources/wsdm18caser.pdf) _WSDM_
- 关键词：Recommender System; Sequential Prediction; Convolutional Neural Networks
- 推荐系统已经成为许多应用的核心技术。大多数系统，如top-N推荐，都会根据用户的一般偏好来推荐商品，而不关注商品的新近程度。Top-N顺序推荐对每个用户过去互动的商品序列进行建模，并旨在预测用户在“不远的将来”可能会进行交互的排名Top-N的商品。交互的顺序意味着，当序列中最近的商品对下一个商品有更大的影响时，序列模式起着重要的作用。本文提出了一个卷积序列嵌入推荐模型(Caser)作为解决这一需求的解决方案。其想法是将最近的商品序列嵌入到时间和潜在空间中的“图像”中，并通过卷积过滤器学习序列模式作为图像的局部特征。这种方法提供了统一和灵活的网络结构来去捕获一般偏好和序列模式。
- [Wang S, Tang J, Wang Y, et al. Exploring Hierarchical Structures for Recommender Systems[J]. IEEE Transactions on Knowledge and Data Engineering, 2018.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8246532) _IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING_
- 真实世界推荐系统中的商品展现出一定的分层结构。同样，用户偏好也呈现层次结构。最近的研究表明，结合商品层次或用户偏好可以提高推荐系统的性能。但是，层次结构通常并不显式可用，尤其是那些用户偏好。因此，层次结构的重要性和可用性之间存在差距。
本文提出了一种新的推荐框架IHSR，它可以在用户和商品的层次结构不明确的情况下捕获用户和商品的隐含结构，并扩展了所提出的框架，以便在可用时捕获显式层次结构，从而产生一个统一框架HSR，该框架能够利用隐式和显式层次结构进行推荐。
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
- [Xue H J, Dai X Y, Zhang J, et al. Deep matrix factorization models for recommender systems[J]. static. ijcai. org, 2017.](http://static.ijcai.org/proceedings-2017/0447.pdf) _ijcai_
- 推荐系统通常通过用户-商品评分矩阵、隐式反馈和辅助信息进行个性化推荐。矩阵分解作为协同过滤中最流行的方法，它可以通过用户和商品的相似性，为某个用户进行个性化推荐。为了解决评分的稀疏性，一些辅助信息如社交关系，商品内容和评论文本等加入到MF过程中。早期的推荐系统利用显式的用户-商品评分矩阵信息，然而仅根据观测评分进行推荐是不高效的，隐式的反馈如购买历史和没有观测到的评分等逐渐被应用到推荐系统中。深度学习由于其强大的表示学习能力，也被应用到推荐系统中。
本文提出了一种新的深层神经矩阵分解模型，利用显式评分和隐式反馈信息用于top-N推荐，其他的一些相关方法只利用了显式评分和隐式反馈信息的一种。该模型利用神经网络将用户和商品非线性映射到一个普通低维空间。使用一个包含显式评分和非偏好隐式反馈的矩阵作为模型的输入。同时设计了一个新的损失函数来考虑明确的评分和隐式反馈，进行更好的优化。
- [Dong X, Yu L, Wu Z, et al. A Hybrid Collaborative Filtering Model with Deep Structure for Recommender Systems[C]//AAAI. 2017: 1309-1315.](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14676/13916) _AAAI_
- 从过去到现在，实际的推荐系统中用的方法可以分为两大类：基于协同过滤和基于内容的过滤方法。协同过滤主要面临着两大困境：1.打分矩阵数据稀疏；2.冷启动问题。目前的工作主要是通过正则化方法，把一些辅助信息融合进矩阵分解的学习过程中，然而学习到的隐特征表达通常是无效的，尤其当评分矩阵和辅助信息很稀疏时，。随着深度学习，尤其是它在表示学习方面的优异性能，这种方法也被融入推荐系统的协同过滤方法中。最近又兴起了一种Bayesian stacked denoising auto-encoder （SDAE）方法，但需要手动设计许多超参。
本文在SDAE的基础上提出一种扩展的 additional SDAE，更为充分地利用辅助信息中有价值的部分，从而缓解冷启动和数据稀疏的问题；提出一种混合式模型，综合了深度学习模型和矩阵分解模型，能够从外部辅助信息和传统的打分信息中学习到更为有效地特征表示。
## 2016 papers
- [Song Y, Elkahky A M, He X. Multi-rate deep learning for temporal recommendation[C]//Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2016: 909-912.](http://delivery.acm.org/10.1145/2920000/2914726/p909-song.pdf?ip=202.113.176.146&id=2914726&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EE4E04C281054793F%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1530646781_035154603033357daaa028281db1643d) _SIGIR_
- 在推荐系统中建模时间行为是一个重要而具有挑战性的问题。它的挑战来自于时间建模增加了参数估计和推理的成本，同时需要大量的数据来可靠地学习模型和额外的时间维度。因此，在大型的现实世界推荐系统中，通常很难对时间行为进行建模。
本文提出了一种新的基于时间深度神经网络的体系结构，该体系结构将长期静态和短期用户偏好相结合，以提高推荐性能。为了对大型应用程序进行有效的训练，本文还提出了一种新颖的预训练方法，可以显著减少自由参数的数量。
- [Wu Y, DuBois C, Zheng A X, et al. Collaborative denoising auto-encoders for top-n recommender systems[C]//Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016: 153-162.WSDM’16, February 22–25, 2016, San Francisco, CA, USA.](http://alicezheng.org/papers/wsdm16-cdae.pdf) _WSDM_
- 近年来，推荐系统已广泛应用于各行各业。给定一组用户、物品和观察到的用户物品交互，这些系统可以给用户推荐其他可能喜欢的物品。个性化推荐是机器学习在电子商务等领域的重要应用之一，许多推荐系统使用协同过滤(CF)方法来进行推荐，因此，top-N推荐方法更值得关注。
本文提出了一个新的基于模型的协同过滤方法Collaborative Denoising Auto-Encoder (CDAE)，Denoising Auto-Encoder可以通过隐层学习到更加鲁棒的特征。作者认为观察到的用户-物品交互，是用户对全体物品偏好的一个“损坏” (corrupted) 版本。利用Auto-Encoder的特点，模型学习根据已观察到的用户-物品偏好，可以重建出用户对所有物品的偏好。
