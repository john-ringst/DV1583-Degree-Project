# DV1583 Degree Project
How can advanced Generative Adversarial Networks (GANs), like WGAN, be used to generate
high-quality, synthetic network traffic and attack data to improve the training and robustness of Collective Intelligence
anomaly detection models?

# Dependencies (Packaged in a Conda environment):
  - python=3.12
  - numpy[version='<2']
  - ipykernel
  - seaborn
  - tqdm
  - matplotlib
  - pandas
  - jupyter
  - scipy
  - torchaudio
  - pytorch
  - torchvision
  - pytorch-cuda=11.8
  - scikit-learn
  - joblib
  - xgboost

  Experiments run on Windows 11, GPU: Nvidia GeForce 3060 TI, CPU: Ryzen 5 5600X

# Acknowledgements
WGAN-GP implementation largely based on implementation by Gulrajani et al. (https://arxiv.org/abs/1704.00028) and Zeleni9 (https://github.com/Zeleni9/pytorch-wgan)