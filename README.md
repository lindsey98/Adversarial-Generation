# Network_Signal
Investigate the distribution of signal

# Background
We believe that for normal examples and adversarial examples. The triggering patterns for internal NN layer are significantly different. In this repository, we aim to verify this hypothesis. 

# Project Structure
```

|
| utils.py: functions that enable partial model execution
| compare_distribution: estimate distribution by Gaussian KDE and compute the KL divergence between two distributions
| dataloader: load training/testing data from CIFAR10
| train.py: train a reference model
|
|___ models: different model architecture
|___ attack: FGSM/Step-LL, DeepFool, JSMA, C&W L2 attack


```

# Related Work

Please refer to the summary.docx for details

- Li, Xin, and Fuxin Li. "Adversarial examples detection in deep networks with convolutional filter statistics." Proceedings of the IEEE International Conference on Computer Vision. 2017. 
- Lu, Jiajun, Theerasit Issaranon, and David Forsyth. "Safetynet: Detecting and rejecting adversarial examples robustly." Proceedings of the IEEE International Conference on Computer Vision. 2017.
- Metzen, Jan Hendrik, et al. "On detecting adversarial perturbations." arXiv preprint arXiv:1702.04267 (2017).
- Gong, Zhitao, Wenlu Wang, and Wei-Shinn Ku. "Adversarial and clean data are not twins." arXiv preprint arXiv:1704.04960(2017).
- Grosse, Kathrin, et al. "On the (statistical) detection of adversarial examples." arXiv preprint arXiv:1702.06280(2017).
- Feinman, Reuben, et al. "Detecting adversarial samples from artifacts." arXiv preprint arXiv:1703.00410 (2017).
- Meng, Dongyu, and Hao Chen. "Magnet: a two-pronged defense against adversarial examples." Proceedings of the 2017 ACM SIGSAC conference on computer and communications security. 2017.

