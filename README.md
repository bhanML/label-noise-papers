# Label Noise Papers

This repository contains Label-Noise Representation Learning (LNRL) papers mentioned in our survey "A Survey of Label-noise Representation Learning: Past, Present, and Future". 

We will update this paper list to include new LNRL papers periodically. 
The current version is updated on 2021.03.03.

## Citation 

Please cite our paper if you find it helpful.

```
@article{han2020survey,
  title={A survey of label-noise representation learning: Past, present and future},
  author={Han, Bo and Yao, Quanming and Liu, Tongliang and Niu, Gang and Tsang, Ivor W and Kwok, James T and Sugiyama, Masashi},
  journal={arXiv preprint arXiv:2011.04406},
  year={2020}
}
```


## Content
1. [Survey](#Survey)
2. [Data](#Data)
    1. [Transition Matrix](#Transition-Matrix)
    1. [Others](#Others)
3. [Objective](#Objective)
    1. [Reguarization](#Reguarization)
    1. [Reweighting](#Reweighting)
    1. [Redesigning](#Redesigning)
    1. [Others](#Others)
4. [Optimization](#Optimization)
    1. [Self-training](#Self-training)
    1. [Co-training](#Co-training)
    1. [Others](#Others)
5. [Future Directions](#Future-Directions)
    1. [New Datasets](#New-Datasets)
    1. [Instance-dependent LNRL](#Instance-dependent-LNRL)
    1. [Adversarial LNRL](#Adversarial-LNRL)
    1. [Noisy Data](#Noisy_Data)


## [Survey](#content)


1. B. Frénay and M. Verleysen, **Classification in the presence of label noise: a survey**, IEEE Transactions on Neural Networks and Learning Systems, vol. 25, no. 5, pp. 845–869, 2013.
[paper](https://romisatriawahono.net/lecture/rm/survey/machine%20learning/Frenay%20-%20Classification%20in%20the%20Presence%20of%20Label%20Noise%20-%202014.pdf)

1. G. Algan and I. Ulusoy, **Image classification with deep learning in the presence of noisy labels: A survey**, arXiv preprint arXiv:1912.05170, 2019.
[paper](https://arxiv.org/pdf/1912.05170.pdf)

1. D. Karimi, H. Dou, S. K. Warfield, and A. Gholipour, **Deep learning with noisy labels: exploring techniques and remedies in medical image analysis**, Medical Image Analysis, 2020.
[paper](https://arxiv.org/pdf/1912.02911.pdf)

1. H. Song, M. Kim, D. Park, and J.-G. Lee, **Learning from noisy labels with deep neural networks: A survey**, arXiv preprint arXiv:2007.08199, 2020.
[paper](https://arxiv.org/pdf/2007.08199.pdf)

## [Data](#content)

### Transition Matrix

1. B. van Rooyen and R. C. Williamson, **A theory of learning with corrupted labels**, Journal of Machine Learning Research, vol. 18, no. 1, pp. 8501–8550, 2017.
[paper](https://www.jmlr.org/papers/volume18/16-315/16-315.pdf)

1. S. Sukhbaatar, J. Bruna, M. Paluri, L. Bourdev, and R. Fergus, **Training convolutional networks with noisy labels**, in ICLR Workshop, 2015.
[paper](https://arxiv.org/pdf/1406.2080.pdf)

1. J. Goldberger and E. Ben-Reuven, **Training deep neural-networks using a noise adaptation layer**, in ICLR, 2017.
[paper](https://openreview.net/pdf?id=H12GRgcxg)

1. G. Patrini, A. Rozza, A. Krishna Menon, R. Nock, and L. Qu, **Making deep neural networks robust to label noise: A loss correction approach**, in CVPR, 2017, pp. 1944–1952.
[paper](https://arxiv.org/pdf/1609.03683.pdf)

1. D. Hendrycks, M. Mazeika, D. Wilson, and K. Gimpel, **Using trusted data to train deep networks on labels corrupted by severe noise**, in NeurIPS, 2018, pp. 10 456–10 465.
[paper](https://arxiv.org/pdf/1802.05300.pdf)

1. M. Lukasik, S. Bhojanapalli, A. K. Menon, and S. Kumar, **Does label smoothing mitigate label noise?** in ICML, 2020.
[paper](https://arxiv.org/pdf/2003.02819.pdf)

1. B. Han, J. Yao, G. Niu, M. Zhou, I. Tsang, Y. Zhang, and M. Sugiyama, **Masking: A new perspective of noisy supervision**, in NeurIPS, 2018, pp. 5836–5846.
[paper](https://arxiv.org/pdf/1805.08193.pdf)

1. X. Xia, T. Liu, N.Wang, B. Han, C. Gong, G. Niu, and M. Sugiyama, **Are anchor points really indispensable in label-noise learning?** in NeurIPS, 2019.
[paper](https://arxiv.org/pdf/1906.00189.pdf)

### Others

1. I. Misra, C. Lawrence Zitnick, M. Mitchell, and R. Girshick, **Seeing through the human reporting bias: Visual classifiers from noisy human-centric labels**, in CVPR, 2016.
[paper](https://arxiv.org/pdf/1512.06974.pdf)

1. J. Krause, B. Sapp, A. Howard, H. Zhou, A. Toshev, T. Duerig, J. Philbin, and L. Fei-Fei, **The unreasonable effectiveness of noisy data for fine-grained recognition**, in ECCV, 2016.
[paper](https://arxiv.org/pdf/1511.06789.pdf)

1. Y. Li, J. Yang, Y. Song, L. Cao, J. Luo, and L.-J. Li, **Learning from noisy labels with distillation**, in ICCV, 2017.
[paper](https://arxiv.org/pdf/1703.02391.pdf)

1. C. G. Northcutt, T.Wu, and I. L. Chuang, **Learning with confident examples: Rank pruning for robust classification with noisy labels,** in UAI, 2017.
[paper](https://arxiv.org/pdf/1705.01936.pdf)

1. Y. Kim, J. Yim, J. Yun, and J. Kim, **Nlnl: Negative learning for noisy labels**, in ICCV, 2019.
[paper](https://arxiv.org/pdf/1908.07387.pdf)

1. P. H. Seo, G. Kim, and B. Han, **Combinatorial inference against label noise**, in NeurIPS, 2019.
[paper](https://papers.nips.cc/paper/2019/file/0cb929eae7a499e50248a3a78f7acfc7-Paper.pdf)

1. T. Kaneko, Y. Ushiku, and T. Harada, **Label-noise robust generative adversarial networks**, in CVPR, 2019.
[paper](https://arxiv.org/pdf/1811.11165.pdf)

1. A. Lamy, Z. Zhong, A. K. Menon, and N. Verma, **Noise-tolerant fair classification**, in NeurIPS, 2019.
[paper](https://proceedings.neurips.cc/paper/2019/file/8d5e957f297893487bd98fa830fa6413-Paper.pdf)

1. J. Yao, H. Wu, Y. Zhang, I. W. Tsang, and J. Sun, **Safeguarded dynamic label regression for noisy supervision**, in AAAI, 2019.
[paper](https://arxiv.org/pdf/1903.02152.pdf)

## [Objective](#content)

### Regularization

1. S. Azadi, J. Feng, S. Jegelka, and T. Darrell, **Auxiliary image regularization for deep cnns with noisy labels**, in ICLR, 2016.
[paper](https://arxiv.org/pdf/1511.07069.pdf)

1. D.-H. Lee, **Pseudo-label: The simple and efficient semisupervised learning method for deep neural networks**, in ICML Workshop, 2013.

1. S. Reed, H. Lee, D. Anguelov, C. Szegedy, D. Erhan, and A. Rabinovich, **Training deep neural networks on noisy labels with bootstrapping**, in ICLR Workshop, 2015.
[paper](https://arxiv.org/pdf/1412.6596.pdf)

1. H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, **mixup: Beyond empirical risk minimization**, in ICLR, 2018.
[paper](https://arxiv.org/pdf/1710.09412.pdf)

1. T. Miyato, S.-i. Maeda, M. Koyama, and S. Ishii, **Virtual adversarial training: a regularization method for supervised and semi-supervised learning**, IEEE transactions on pattern analysis and machine intelligence, vol. 41, no. 8, pp. 1979–1993, 2018.
[paper](https://arxiv.org/pdf/1704.03976.pdf)

1. B. Han, G. Niu, X. Yu, Q. Yao, M. Xu, I. Tsang, and M. Sugiyama, **Sigua: Forgetting may make learning with noisy labels more robust**, in ICML, 2020.
[paper](https://arxiv.org/pdf/1809.11008.pdf)

### Reweighting

1. T. Liu and D. Tao, **Classification with noisy labels by importance reweighting**, IEEE Transactions on pattern analysis and machine intelligence, vol. 38, no. 3, pp. 447–461, 2015.
[paper](https://arxiv.org/pdf/1411.7718.pdf)

1. Y. Wang, A. Kucukelbir, and D. M. Blei, **Robust probabilistic modeling with bayesian data reweighting,** in ICML, 2017, pp. 3646–3655.
[paper](https://arxiv.org/pdf/1606.03860.pdf)

1. E. Arazo, D. Ortego, P. Albert, N. E. O’Connor, and K. McGuinness, **Unsupervised label noise modeling and loss correction**, in ICML, 2019.
[paper](https://arxiv.org/pdf/1904.11238.pdf)

1. J. Shu, Q. Xie, L. Yi, Q. Zhao, S. Zhou, Z. Xu, and D. Meng, “Meta-weight-net: Learning an explicit mapping for sample weighting,” in NeurIPS, 2019, pp. 1919–1930.
[paper](https://arxiv.org/pdf/1902.07379.pdf)

### Redesigning

1. A. K. Menon, A. S. Rawat, S. J. Reddi, and S. Kumar, **Can gradient clipping mitigate label noise?** in ICLR, 2020.
[paper](https://openreview.net/pdf?id=rklB76EKPr)

1. Z. Zhang and M. Sabuncu, **Generalized cross entropy loss for training deep neural networks with noisy labels**, in NeurIPS, 2018, pp. 8778–8788.
[paper](https://arxiv.org/pdf/1805.07836.pdf)

1. N. Charoenphakdee, J. Lee, and M. Sugiyama, **On symmetric losses for learning from corrupted labels**, in ICML, 2019.
[paper](http://proceedings.mlr.press/v97/charoenphakdee19a/charoenphakdee19a.pdf)

1. S. Thulasidasan, T. Bhattacharya, J. Bilmes, G. Chennupati, and J. Mohd-Yusof, **Combating label noise in deep learning using abstention**, in ICML, 2019.
[paper](https://arxiv.org/pdf/1905.10964.pdf)

1. Y. Lyu and I. W. Tsang, **Curriculum loss: Robust learning and generalization against label corruption**, in ICLR, 2020.
[paper](https://arxiv.org/pdf/1905.10045.pdf)

1. S. Laine and T. Aila, **Temporal ensembling for semi-supervised learning,** in ICLR, 2017.
[paper](https://arxiv.org/pdf/1610.02242.pdf)

1. D. T. Nguyen, C. K. Mummadi, T. P. N. Ngo, T. H. P. Nguyen, L. Beggel, and T. Brox, **Self: Learning to filter noisy labels with self-ensembling**, in ICLR, 2020.
[paper](https://arxiv.org/pdf/1910.01842.pdf)

1. X. Ma, Y. Wang, M. E. Houle, S. Zhou, S. M. Erfani, S.-T. Xia, S. Wijewickrema, and J. Bailey, **Dimensionality-driven learning with noisy labels**, in ICML, 2018.
[paper](http://proceedings.mlr.press/v80/ma18d/ma18d.pdf)

### Others

1. S. Branson, G. Van Horn, and P. Perona, **Lean crowdsourcing: Combining humans and machines in an online system**, in CVPR, 2017.
[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Branson_Lean_Crowdsourcing_Combining_CVPR_2017_paper.pdf)

1. A. Vahdat, **Toward robustness against label noise in training deep discriminative neural networks,** in NeurIPS, 2017.
[paper](https://arxiv.org/pdf/1706.00038.pdf%20%C3%A2%E2%82%AC%E2%80%B9)

1. H.-S. Chang, E. Learned-Miller, and A. McCallum, **Active bias: Training more accurate neural networks by emphasizing high variance samples**, in NeurIPS, 2017.
[paper](https://arxiv.org/pdf/1704.07433.pdf)

1. A. Khetan, Z. C. Lipton, and A. Anandkumar, **Learning from noisy singly-labeled data,** ICLR, 2018.
[paper](https://arxiv.org/pdf/1712.04577.pdf)

1. D. Tanaka, D. Ikami, T. Yamasaki, and K. Aizawa, **Joint optimization framework for learning with noisy labels**, in CVPR, 2018.
[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tanaka_Joint_Optimization_Framework_CVPR_2018_paper.pdf)

1. Y. Wang, W. Liu, X. Ma, J. Bailey, H. Zha, L. Song, and S.-T. Xia, **Iterative learning with open-set noisy labels**, in CVPR, 2018.
[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Iterative_Learning_With_CVPR_2018_paper.pdf)

1. S. Jenni and P. Favaro, **Deep bilevel learning**, in ECCV, 2018.
[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Simon_Jenni_Deep_Bilevel_Learning_ECCV_2018_paper.pdf)

1. Y. Wang, X. Ma, Z. Chen, Y. Luo, J. Yi, and J. Bailey, **Symmetric cross entropy for robust learning with noisy labels,** in ICCV, 2019.
[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf)

1. J. Li, Y. Song, J. Zhu, L. Cheng, Y. Su, L. Ye, P. Yuan, and S. Han, **Learning from large-scale noisy web data with ubiquitous reweighting for image classification**, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.
[paper](https://arxiv.org/pdf/1811.00700.pdf)

1. Y. Xu, P. Cao, Y. Kong, and Y. Wang, **L_dmi: A novel informationtheoretic loss function for training deep nets robust to label noise**, in NeurIPS, 2019.
[paper](https://openreview.net/pdf/14f442968372d127473b832165df3e78abc7a1db.pdf)

1. Y. Liu and H. Guo, **Peer loss functions: Learning from noisy labels without knowing noise rates**, in ICML, 2020.
[paper](http://proceedings.mlr.press/v119/liu20e/liu20e.pdf)

1. X. Ma, H. Huang, Y. Wang, S. Romano, S. Erfani, and J. Bailey, **Normalized loss functions for deep learning with noisy labels**, in ICML, 2020.
[paper](http://proceedings.mlr.press/v119/ma20c/ma20c.pdf)

## [Optimization](#content)

### Self-training

1. L. Jiang, Z. Zhou, T. Leung, L.-J. Li, and L. Fei-Fei, **Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels**, in ICML, 2018, pp. 2304–2313.
[paper](http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf)

1. M. Ren, W. Zeng, B. Yang, and R. Urtasun, **Learning to reweight examples for robust deep learning**, in ICML, 2018.
[paper](http://proceedings.mlr.press/v80/ren18a/ren18a.pdf)

1. L. Jiang, D. Huang, M. Liu, W. Yang. **Beyond synthetic noise: Deep learning on controlled noisy labels**, in ICML 2020.
[paper](http://proceedings.mlr.press/v119/jiang20c/jiang20c.pdf)

### Co-training

1. B. Han, Q. Yao, X. Yu, G. Niu, M. Xu, W. Hu, I. Tsang, and M. Sugiyama, **Co-teaching: Robust training of deep neural networks with extremely noisy labels**, in NeurIPS, 2018, pp. 8527–8537.
[paper](https://arxiv.org/pdf/1804.06872.pdf)

1. X. Yu, B. Han, J. Yao, G. Niu, I. W. Tsang, and M. Sugiyama, **How does disagreement help generalization against label corruption?** in ICML, 2019.
[paper](http://proceedings.mlr.press/v97/yu19b/yu19b.pdf)

1. Q. Yao, H. Yang, B. Han, G. Niu, and J. T. Kwok, **Searching to exploit memorization effect in learning with noisy labels**, in ICML, 2020.
[paper](http://proceedings.mlr.press/v119/yao20b/yao20b.pdf)

### Others

1. J. Li, R. Socher, and S. C. Hoi, **Dividemix: Learning with noisy labels as semi-supervised learning**, in ICLR, 2020.
[paper](https://arxiv.org/pdf/2002.07394.pdf)

1. D. Hendrycks, K. Lee, and M. Mazeika, **Using pre-training can improve model robustness and uncertainty**, in ICML, 2019.
[paper](http://proceedings.mlr.press/v97/hendrycks19a/hendrycks19a.pdf)

1. D. Bahri, H. Jiang, and M. Gupta, **Deep k-nn for noisy labels**, in ICML, 2020.
[paper](http://proceedings.mlr.press/v119/bahri20a/bahri20a.pdf)

1. P. Chen, B. Liao, G. Chen, and S. Zhang, **Understanding and utilizing deep neural networks trained with noisy labels**, in ICML, 2019.
[paper](http://proceedings.mlr.press/v97/chen19g/chen19g.pdf)

## [Future Directions](#content)

### New Datasets

1. T. Xiao, T. Xia, Y. Yang, C. Huang, and X. Wang, **Learning from massive noisy labeled data for image classification**, in CVPR, 2015, pp. 2691–2699.
[paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf)

1. L. Jiang, D. Huang, M. Liu, and W. Yang, **Beyond synthetic noise: Deep learning on controlled noisy labels**, in ICML, 2020.
[paper](http://proceedings.mlr.press/v119/jiang20c/jiang20c.pdf)

### Instance-dependent LNRL

1. A. Menon, B. Van Rooyen, and N. Natarajan, **Learning from binary labels with instance-dependent corruption**, Machine Learning, vol. 107, p. 1561–1595, 2018.
[paper](https://arxiv.org/pdf/1605.00751.pdf)

1. J. Cheng, T. Liu, K. Ramamohanarao, and D. Tao, **Learning with bounded instance-and label-dependent label noise**, in ICML, 2020.
[paper](http://proceedings.mlr.press/v119/cheng20c/cheng20c.pdf)

1. A. Berthon, B. Han, G. Niu, T. Liu, and M. Sugiyama, **Confidence scores make instance-dependent label-noise learning possible**, arXiv preprint arXiv:2001.03772, 2020.
[paper](https://arxiv.org/pdf/2001.03772.pdf)

### Adversarial LNRL

1. Y. Wang, D. Zou, J. Yi, J. Bailey, X. Ma, and Q. Gu, **Improving adversarial robustness requires revisiting misclassified examples**, in ICLR, 2020.
[paper](https://openreview.net/pdf?id=rklOg6EFwS)

1. J. Zhang, X. Xu, B. Han, G. Niu, L. Cui, M. Sugiyama, and M. Kankanhalli, **Attacks which do not kill training make adversarial learning stronger**, in ICML, 2020.
[paper](http://proceedings.mlr.press/v119/zhang20z/zhang20z.pdf)

### Noisy Data

1. J. Zhang, B. Han, L. Wynter, K. H. Low, and M. Kankanhalli, **Towards robust resnet: A small step but a giant leap**, in IJCAI, 2019.
[paper](https://arxiv.org/pdf/1902.10887.pdf)

1. B. Han, Y. Pan, and I. W. Tsang, **Robust plackett–luce model for k-ary crowdsourced preferences**, Machine Learning, vol. 107, no. 4, pp. 675–702, 2018.
[paper](https://link.springer.com/content/pdf/10.1007/s10994-017-5674-0.pdf)

1. Y. Pan, B. Han, and I.W. Tsang, **Stagewise learning for noisy k-ary preferences**, Machine Learning, vol. 107, no. 8-10, pp. 1333–1361, 2018.
[paper](https://link.springer.com/content/pdf/10.1007/s10994-018-5716-2.pdf)

1. F. Liu, J. Lu, B. Han, G. Niu, G. Zhang, and M. Sugiyama, **Butterfly: A panacea for all difficulties in wildly unsupervised domain adaptation**, arXiv preprint arXiv:1905.07720, 2019.
[paper](https://arxiv.org/pdf/1905.07720.pdf)

1.  X. Yu, T. Liu, M. Gong, K. Zhang, K. Batmanghelich, and D. Tao, **Label-noise robust domain adaptation**,” in ICML, 2020.
[paper](http://proceedings.mlr.press/v119/yu20c/yu20c.pdf)

1. S. Wu, X. Xia, T. Liu, B. Han, M. Gong, N. Wang, H. Liu, and G. Niu, **Multi-class classification from noisy-similarity-labeled data**, arXiv preprint arXiv:2002.06508, 2020.
[paper](https://arxiv.org/pdf/2002.06508.pdf)

1. C. Wang, B. Han, S. Pan, J. Jiang, G. Niu, and G. Long, **Crossgraph: Robust and unsupervised embedding for attributed graphs with corrupted structure**, in ICDM, 2020.
[paper]()

1. Y.-H. Wu, N. Charoenphakdee, H. Bao, V. Tangkaratt, and M. Sugiyama, **Imitation learning from imperfect demonstration**, in ICML, 2019.
[paper](https://arxiv.org/pdf/1901.09387.pdf)

1. D. S. Brown, W. Goo, P. Nagarajan, and S. Niekum, **Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations**, in ICML, 2019.
[paper](https://arxiv.org/pdf/1904.06387.pdf)

1. J. Audiffren, M. Valko, A. Lazaric, and M. Ghavamzadeh, **Maximum entropy semi-supervised inverse reinforcement learning**, in IJCAI, 2015.
[paper](https://hal.inria.fr/hal-01146187/document)

1. V. Tangkaratt, B. Han, M. E. Khan, and M. Sugiyama, **Variational imitation learning with diverse-quality demonstrations**, in ICML, 2020.
[paper](https://pdfs.semanticscholar.org/f319/069e750f7178727b7e161570d036ca34a082.pdf)

### Automated Machine Learning (AutoML)

1. Q. Yao, H. Yang, B. Han, G. Niu, J. Kwok. **Searching to exploit memorization effect in learning from noisy labels**, ICML, 2020, [paper](http://arxiv.org/abs/1911.02377) [code](https://github.com/AutoML-4Paradigm/S2E)
