# Knowledge-assistance Knowledge-mining Kongledge-debiased Framework for Multi-domain Fake News Detection(K3MDFEND)

This is a concrete implementation of our paper"Knowledge-assistance Knowledge-mining Kongledge-debiased Framework for Multi-domain Fake News Detection" which is submitting to IPM 2025.5.

With the rapid growth of social media, news events increasingly exhibit cross-domain topicality, making multi-domain fake news detection a critical yet challenging task. A key difficulty lies in balancing model performance improvement with bias reduction. To address this, we propose the Knowledge-assistance Knowledge-mining Knowledge-debiased Multi-domain Fake News Detection Framework (K3MDFEND). To obtain more robust feature representations and reduce domain bias, we integrate Large Language Models (LLMs) through a novel argumentation-based prompt engineering framework to avoid over-focusing on the capture of domain-specific features. This generates reliable external knowledge by simulating role-specific debates, which is then fused with comment features via a Quality-Aware Attention Fusion module that dynamically weights evidence credibility. To further distill key insights from the comments while preserving their inherent semantic integrity, we leverage feature alignment techniques on the comment features. By integrating comment features, we successfully reduced the reliance on domain-invariant features, thereby decreasing domain bias. We optimize traditional contrastive learning to prioritize domain-specific knowledge acquisition, effectively mitigating domain bias. Extensive experiments on Chinese and English datasets demonstrate that K3MDFEND achieves state-of-the-art performance in both detection performance and bias metric reduction, outperforming existing multi-domain debiasing methods.

## Introduction

This repository provides specific implementations of the K3MDFEND framework and 14 baselines, our backbone is from the DTDBD framework in ICDE 2024.

## Requirements

Python 3.8

PyTorch>1.0

Pandas

Numpy

Tqdm

Transformers

## Run

dataset: the English(en) or Chinese dataset(ch1)

gpu: the index of gpu you will use, default for `0`

model:./models/kairos

model_name: model_name within textcnn bigru bert eann eddfn mmoe mose dualemotion stylelstm mdfend m3fend.

for DTDBD running, you need to click this link: https://github.com/ningljy/DTDBD/tree/main

Please note that you need to train both two teacher models, the way to do that you need to read DTDBD paper.

You can run this code through to train baseline model:

```
python mainCDK.py --gpu 0 --lr 0.0001 --dataset ch1

```
## Reference
