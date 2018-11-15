# NLP  Datasets


## Sentiment

### [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/): 

An older, relatively small dataset for binary sentiment classification features 25,000 movie reviews.

"用于二元情感分类的较旧的，相对较小的电影评论数据集。"

- statistics

| Size | #Num | Size |
| ------ | ------ | ------ |
| xxx | xx | xxx |
    
- Examples
    
    

### [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/code.html): 

Standard sentiment dataset with sentiment annotations.

"具有情感注释的标准情绪数据集。"



### [Sentiment140](http://help.sentiment140.com/for-students/)

A popular dataset, which uses 160,000 tweets with emoticons pre-removed.

"一个预先删除表情符号的包含160,000条推文的数据集。"

### [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

Twitter data on US airlines from February 2015, classified as positive, negative, and neutral tweets.

"2015年2月的美国航空公司的Twitter数据，标注为正面，负面和中性。"


### Product review

### [Yelp Reviews](https://www.yelp.com/dataset)

An open dataset released by Yelp, contains more than 5 million reviews.

"Yelp发布的开放数据集,包含超过500万条评论。"

### [Amazon Reviews](https://snap.stanford.edu/data/web-Amazon.html)

Contains around 35 million reviews from Amazon spanning 18 years. Data include product and user information, ratings, and the plaintext review.

"包含18年来亚马逊商城的约3500万条评论。数据包括产品和用户信息，评级和文本评论。"


## Summary


## Machine Translation


## Question Answering

### [HotspotQA Dataset](https://hotpotqa.github.io/)

Question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.

"以自然，多跳问题为特色的问答数据集，具有支持事实的强大监督，以实现更易于解释的问答系统。"


## Image

### [MS COCO](http://cocodataset.org/) 

 Generic image understanding and captioning.
 
 "带文本标注的图片数据集"

- statistics (2017-version)

| Train | Val | Test |
| ------ | ------ | ------ |
| 118K/18GB | 5K/1GB | 41K/6GB |
    
- Examples

 
### [Labelme]()

A large dataset of annotated images. Using matlab tools

"带标注的大型图像数据集.(matlab下使用?)"



### [ImageNet](http://image-net.org/)

The de-facto image dataset for new algorithms, organized according to the WordNet hierarchy, in which hundreds and thousands of images depict each node of the hierarchy.
    





### [COIL100](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)
 
100 different objects imaged at every angle in a 360 rotation.

"100个不同的物体在360度旋转的每个角度成像。"


### [LSUN](http://lsun.cs.princeton.edu/2016/)

Scene understanding with many ancillary tasks (room layout estimation, saliency prediction, etc.)

"场景理解与许多辅助任务（房间布局估计，显着性预测等）"


## Embeddings

### [word2vec-GoogleNews-vectors](https://code.google.com/archive/p/word2vec/)

A Google News Corpus (3 billion running words) word vector model (3 million 300-dimensional English word vectors) pre-trained with the word2vec tool.

"使用word2vec工具预先训练的Google新闻语料库300维英语单词向量"

Mirror site: [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)



### [CA8-Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors)

This project provides 100+ Chinese Word Vectors (embeddings) trained with different representations (dense and sparse), context features (word, ngram, character, and more), and corpora.

"该项目提供了100多个使用不同表示（密集和稀疏），上下文特征（单词，ngram，字符等）和语料库训练的中文单词向量（嵌入）。"


### [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/embedding.html)

This corpus provides 200-dimension vector representations, a.k.a. embeddings, for over 8 million Chinese words and phrases, which are pre-trained on large-scale high-quality data. 

"该语料库为超过800万个中文单词和短语提供了200维词向量表示"


## Others


### [Google Books Ngrams](https://aws.amazon.com/datasets/google-books-ngrams/)

A collection of words from Google books.

"统计自Google Books的词集合。"


### [Rumors in Chinese social media](https://github.com/thunlp/Chinese_Rumor_Dataset)

"Chinese rumor data, which was crawled from Sina Weibo's false information reporting platform, with a total of 31,669 as of June 13, 2017."

“中文谣言数据，该数据为从新浪微博不实信息举报平台抓取的中文谣言数据，共包含截止至2017年6月13日的31669条谣言。”

- Examples(Each instance contains the following information.)

    - rumorCode: 该条谣言的唯一编码，可以通过该编码直接访问该谣言举报页面。
    - title: 该条谣言被举报的标题内容
    - informerName: 举报者微博名称
    - informerUrl: 举报者微博链接
    - rumormongerName: 发布谣言者的微博名称
    - rumormongerUr: 发布谣言者的微博链接
    - rumorText: 谣言内容
    - visitTimes: 该谣言被访问次数
    - result: 该谣言审查结果
    - publishTime: 该谣言被举报时间

---
## References

Thanks to the following articles!

[1][The 50 Best Public Datasets for Machine Learning](https://medium.com/datadriveninvestor/the-50-best-public-datasets-for-machine-learning-d80e9f030279)
