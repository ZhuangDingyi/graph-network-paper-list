# graph-network-paper-list
This is [my](https://zhuangdingyi.github.io/) paper reading list concerning current interesting graph network problems. Papers are classified by methods and sorted by descending-year orders. The format of each paper follows **Title (Journal/Conference/Review_forum Year)**. Particularly, I will have preference on how graph network can help solve spatial-temporal problems. You are highly welcome to help complete the paper reading list.
<!-- TOC -->

- [graph-network-paper-list](#graph-network-paper-list)
- [General review](#general-review)
- [Graph Convolution Network (GCN)](#graph-convolution-network-gcn)
  - [Fundamental: learn graph in spectral domain](#fundamental-learn-graph-in-spectral-domain)
  - [Spatial-temporal GCN](#spatial-temporal-gcn)
  - [Dynamic GCN/GNN](#dynamic-gcngnn)
  - [GCN for Directed Graph](#gcn-for-directed-graph)
- [Graph Recurrent Neural Networks](#graph-recurrent-neural-networks)
- [Graph Embedding and Graph Representation Learning](#graph-embedding-and-graph-representation-learning)
  - [Survey](#survey)
  - [Embedding nodes](#embedding-nodes)
  - [Embedding sub-graphs](#embedding-sub-graphs)
  - [Spatial-temporal graph embedding](#spatial-temporal-graph-embedding)
- [Diffusion Graph](#diffusion-graph)
- [Graph Attention (GAT)](#graph-attention-gat)
- [Graph Kernels](#graph-kernels)
- [Generative Graph](#generative-graph)
- [Library](#library)
- [Interesting groups and topics](#interesting-groups-and-topics)

<!-- /TOC -->

# General review
- [Graph Neural Networks: A Review of Methods and Applications (arXiv 2019)](https://arxiv.org/pdf/1812.08434.pdf)
- [How Powerfl are graph neural networks (arXiv 2018)](https://arxiv.org/pdf/1810.00826.pdf)

- [Geometric deep learning: going beyond Euclidean data (IEEE Signal Processing Magazine 2017)](https://arxiv.org/pdf/1611.08097.pdf)

- [Convolutional Networks on Graphs for Learning Molecular Fingerprints (NeurIPS 2015)](https://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints.pdf)

- [The graph neural network model (IEEE TRANSACTIONS ON NEURAL NETWORKS 2009)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1015.7227&rep=rep1&type=pdf)

# Graph Convolution Network (GCN)
## Fundamental: learn graph in spectral domain
- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeurIPS 2016)](https://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf)
- [Learning Convolutional Neural Networks for Graphs (ICML 2016)](http://proceedings.mlr.press/v48/niepert16.pdf)
- [The emerging field of signal processing on graphs- Extending high-dimensional data analysis to networks and other irregular domains (IEEE Signal Processing Magazine 2013)](https://ieeexplore.ieee.org/document/6494675)
## Spatial-temporal GCN
- [3D Graph Convolutional Networks with Temporal Graphs: A Spatial Information Free Framework For Traffic Forecasting (arXiv 2019)](https://arxiv.org/pdf/1903.00919.pdf)
- [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (AAAI 2019)](https://aaai.org/ojs/index.php/AAAI/article/view/3881)
- [Multi-Modal Graph Interaction for Multi-Graph Convolution Network in Urban Spatiotemporal Forecasting (arXiv 2019)](https://arxiv.org/abs/1905.11395)
- [Graph Convolutional Neural Networks for Human Activity Purpose Imputation from GPS-based Trajectory Data (Openreview 2018)](https://openreview.net/pdf?id=H1xYUOmy1V)
- [Spatio-Temporal Graph Convolutional Networks-A Deep Learning Framework for Traffic Forecasting (IJCAI 2017)](https://www.ijcai.org/Proceedings/2018/0505.pdf)

## Dynamic GCN/GNN
- [EvolveGCN Evolving Graph Convolutional Networks for Dynamic Graphs (AAAI 2020)](https://arxiv.org/abs/1902.10191)
- [Dynamic spatial-temporal graph convolutional neural network for Traffic Forecasting(AAAI 2019)](https://www.aaai.org/ojs/index.php/AAAI/article/view/3877)
- [Generalizing Graph Convolutional Neural Networks with Edge-Variant Recursions on Graphs (arXiv 2019)](https://arxiv.org/abs/1903.01298)
- [Temporal Link Prediction in Dynamic Networks (MLG Workshop 2019)](https://www.mlgworkshop.org/2019/papers/MLG2019_paper_22.pdf)
- [Link Prediction in Dynamic Weighted and Directed Social Network using Supervised Learning (Surface 2015)](https://surface.syr.edu/cgi/viewcontent.cgi?article=1355&context=etd)
- - [Nonparametric Link Prediction in Dynamic Networks (arXiv 2012)](https://arxiv.org/pdf/1206.6394.pdf)
## GCN for Directed Graph
- [MOTIFNET: A MOTIF-BASED GRAPH CONVOLUTIONAL NETWORK FOR DIRECTED GRAPHS (IEEE Xplore 2018)](https://ieeexplore.ieee.org/document/8439897)

# Graph Recurrent Neural Networks

- [Traffic Graph Convolutional Recurrent Neural Network Deep Learning Framework for Network Scale Traffic Learning and Forecasting (arXiv 2019)](https://arxiv.org/abs/1802.07007)
- [Efficient Metropolitan Traffic Prediction Based on Graph Recurrent Neural Network (arXiv 2018)](https://arxiv.org/abs/1811.00740)


# Graph Embedding and Graph Representation Learning

## Survey
- [Relational inductive biases, deep learning, and graph networks (arXiv 2018)](https://arxiv.org/pdf/1806.01261.pdf)
- [A Comprehensive Survey of Graph Embedding Problems, Techniques and Applications (arXiv 2018)](https://arxiv.org/abs/1709.07604)
- [Representation Learning on Graphs: Methods and Applications (arXiv 2017)](https://arxiv.org/pdf/1709.05584.pdf)

## Embedding nodes

## Embedding sub-graphs

## Spatial-temporal graph embedding

# Diffusion Graph
- [Diffusion Improves Graph Learning (NeurIPS 2019)](https://arxiv.org/abs/1911.05485)
- [Diffusion Convolutional Recurrent Neural Network Data-Driven Traffic Forecasting (arXiv 2018)](https://arxiv.org/abs/1707.01926)

# Graph Attention (GAT)
- [Heterogeneous Graph Attention Network (WWW 2019)](https://arxiv.org/abs/1903.07293)
- [Relational Graph Attention Networks (ICLR 2019)](https://arxiv.org/pdf/1904.05811.pdf)
- [Graph Attention Networks (ICLR 2018)](https://arxiv.org/abs/1710.10903)
- [DeepInf: Social Influence Prediction with Deep Learning (KDD 2018)](https://arxiv.org/abs/1807.05560)
- [Inductive Representation Learning on Large Graphs (NeurIPS 2017)](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- [Neural Message Passing for Quantum Chemistry (arXiv 2017)](https://arxiv.org/abs/1704.01212)
# Graph Kernels
- [A survey on graph kernels (arXiv 2019)](https://arxiv.org/abs/1903.11835)
- [Collective dynamics of ‘small-world’ networks (Nature 1998)](https://www.nature.com/articles/30918)
  
# Generative Graph
- [Generative Graph Convolutional Network for Growing Graphs (ICASSP 2019)](https://ieeexplore.ieee.org/abstract/document/8682360)
- [Efficient Graph Generation with Graph Recurrent Attention Networks (NeurIPS 2019)](http://papers.nips.cc/paper/8678-efficient-graph-generation-with-graph-recurrent-attention-networks)
- [A generative graph model for electrical infrastructure networks (Journal of Complex Networks 2018)](https://academic.oup.com/comnet/article/7/1/128/5073058)
- [GraphGAN Graph Representation Learning With Generative Adversarial Nets (AAAI 2018)](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16611)
- [Graphite: Iterative Generative Modeling of Graphs (arXiv 2018)](https://arxiv.org/pdf/1803.10459.pdf)
- [GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders (ICANN 2018)](https://link.springer.com/chapter/10.1007/978-3-030-01418-6_41)
- [MolGAN: An implicit generative model for small molecular graphs (arXiv 2018)](https://arxiv.org/pdf/1805.11973.pdf)
- [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models (arXiv 2018)](https://arxiv.org/pdf/1802.08773.pdf)
- [Junction Tree Variational Autoencoder for Molecular Graph Generation (arXiv 2018)](https://arxiv.org/pdf/1802.04364.pdf)
- [Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation (NeurIPS 2018)](https://arxiv.org/pdf/1806.02473.pdf)
- [Constrained Graph Variational Autoencoders for Molecule Design (NeurIPS 2018)](http://papers.nips.cc/paper/8005-constrained-graph-variational-autoencoders-for-molecule-design.pdf)
# Library

# Interesting groups and topics
- [Jure Leskovec](https://cs.stanford.edu/people/jure/)
- [William L. Hamilton](https://www.cs.mcgill.ca/~wlh/)