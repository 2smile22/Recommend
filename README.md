# Deep Learning and Recommenders
## 2017 papers
- [Li X, She J. Collaborative Variational Autoencoder for Recommender Systems[C]// The, ACM SIGKDD International Conference. ACM, 2017:305-314.](https://dl.acm.org/citation.cfm?doid=3097983.3098077]) _SIGKDD_
### 现有的推荐系统方法大致可分为三类:基于内容的方法、协同过滤方法和混合方法。协同过滤方法具有稀疏性、冷启动等缺点，混合方法考虑了评分和内容信息。越来越多的人开始关注混合方法，它们同时考虑评分和内容信息，以往的研究大多不能学习推荐任务内容的良好表示，或者只考虑内容的文本模式，因此这些方法在当前的多媒体场景（如文本、图像）中是非常有限的。本文提出了一种贝叶斯生成模型称为collaborative variational autoencoder(CVAE)，该模型考虑了多媒体场景中推荐的评分和内容。模型以一种无监督的方式从内容数据中学习深层次的潜在表达形式，并从内容和评分两个方面学习商品和用户之间的隐式关系。与以往的去噪标准不同的是，所提出的CVAE通过推理网络获取了隐藏空间（非观测空间）中内容的潜在分布，可以很容易地扩展到除文本之外的其他多媒体模式。
- [Li J, Ren P, Chen Z, et al. Neural Attentive Session-based Recommendation[J]. 2017.CIKM’17 , November 6–10, 2017, Singapore.](https://dl.acm.org/citation.cfm?id=3132926) _CIKM_
