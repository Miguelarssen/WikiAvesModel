This repository contains the implementation of my undergraduate thesis titled “An MLOps Architecture for the Lifecycle of Bioacoustic Models: A Case Study Using WikiAves Data”.

The project proposes and implements a Level 2 MLOps architecture to automate the full lifecycle of a deep learning model for Brazilian bird sound classification. The dataset is built through web scraping of a public collaborative platform, WikiAves, which hosts millions of bird records from Brazil. The collected audio data is processed, transformed into spectrograms, and used to train deep learning architectures such as ResNet and CRNN.

The system integrates Continuous Integration, Continuous Training, data and model versioning (DVC and MLflow), automated experimentation, and monitoring for data and model drift. The goal is to demonstrate how MLOps practices can be applied to large-scale bioacoustic data, contributing both to software engineering research and to AI-driven biodiversity monitoring in Brazil.
