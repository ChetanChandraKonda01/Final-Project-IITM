# Final-Project-IITM

# **Sequence-to-Sequence and NLP-Based Projects with CNN Comparison**

This repository contains three diverse projects that demonstrate expertise in Machine Learning, Deep Learning, and Natural Language Processing. Each project leverages advanced techniques to solve specific tasks and highlights the implementation of state-of-the-art models and methods.

---

## **Table of Contents**

1. [Project 1: Sequence-to-Sequence Modeling with Attention Mechanism](#project-1-sequence-to-sequence-modeling-with-attention-mechanism)
2. [Project 2: Multifunctional NLP and Image Generation Tool using Hugging Face Models](#project-2-multifunctional-nlp-and-image-generation-tool-using-hugging-face-models)
3. [Project 3: Comparison of CNN Architectures on Different Datasets](#project-3-comparison-of-cnn-architectures-on-different-datasets)
4. [Technologies Used](#technologies-used)
5. [Setup Instructions](#setup-instructions)
6. [Results Overview](#results-overview)
7. [Contact](#contact)

---

## **Project 1: Sequence-to-Sequence Modeling with Attention Mechanism**

### **Overview**
The goal of this project is to implement a sequence-to-sequence (seq2seq) model with an attention mechanism to predict a target sequence (reversed version of the input sequence). The project highlights the importance of attention in improving model performance.

### **Skills Gained**
- Sequence-to-sequence modeling
- Attention mechanisms
- PyTorch implementation for NLP tasks
- Model evaluation and performance analysis

### **Approach**
1. Synthetic dataset generation: Random sequences of integers where the target is the reverse of the input.
2. Implemented seq2seq model with an attention mechanism using PyTorch.
3. Trained and evaluated the model using loss and accuracy metrics.

### **Business Use Cases**
- Machine Translation Systems
- Text Summarization Tools
- Chatbots and Conversational AI

### **Results**
- Loss curves and model accuracy on synthetic data.
- Demonstration of the attention mechanism’s impact on performance.

---

## **Project 2: Multifunctional NLP and Image Generation Tool using Hugging Face Models**

### **Overview**
This project involves building a user-friendly application that integrates multiple pretrained models from Hugging Face for NLP and computer vision tasks. The tool supports tasks like text summarization, sentiment analysis, question answering, next-word prediction, and image generation.

### **Skills Gained**
- Hugging Face Transformers
- Implementing versatile NLP models
- Backend and frontend integration
- Model performance evaluation

### **Approach**
1. Loaded pretrained models for tasks:
   - Text summarization
   - Chatbot
   - Sentiment analysis
   - Story prediction
   - Image generation
2. Built a user-friendly interface to interact with the models.
3. Evaluated performance using accuracy, F1-score, precision, and recall.

### **Business Use Cases**
- AI tools for content creation
- Customer service enhancement with chatbots
- Creative AI-powered tools for storytelling and image generation

### **Results**
- A fully functional tool integrating multiple NLP tasks with image generation.
- Evaluation metrics showcasing model performance.

---

## **Project 3: Comparison of CNN Architectures on Different Datasets**

### **Overview**
The objective of this project is to compare the performance of popular Convolutional Neural Network (CNN) architectures (LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, and SENet) across three datasets: MNIST, FMNIST, and CIFAR-10.

### **Skills Gained**
- Implementing CNN architectures using PyTorch and TensorFlow
- Comparative analysis of models
- Dataset preprocessing and image classification
- Performance evaluation with loss curves, accuracy, precision, and recall

### **Approach**
1. Loaded and preprocessed MNIST, FMNIST, and CIFAR-10 datasets.
2. Implemented CNN architectures:
   - LeNet-5, AlexNet, GoogLeNet, VGGNet, ResNet, Xception, and SENet.
3. Trained each architecture and evaluated performance using accuracy, precision, recall, and F1-score.

### **Business Use Cases**
- Optimizing resource usage for computer vision tasks
- Selecting the best CNN model for specific datasets and business needs

### **Results**
- Comparative loss and accuracy curves for all CNN architectures.
- Performance metrics for MNIST, FMNIST, and CIFAR-10 datasets.

## Comparison of CNN Architectures on MNIST Dataset

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| **AlexNet** | 0.9922   | 0.9920    | 0.9922 | 0.9921   |
| **GoogLeNet** | 0.9922  | 0.9922    | 0.9921 | 0.9921   |
| **VGG16**   | 0.9925   | 0.9926    | 0.9924 | 0.9925   |
| **Xception** | 0.9656  | 0.9660    | 0.9652 | 0.9654   |
| **LeNet5**  | 0.9900   | 0.9899    | 0.9899 | 0.9899   |
| **ResNet**  | 0.9924   | 0.9926    | 0.9922 | 0.9924   |

### Insights:
- **VGG16** achieved the highest accuracy (0.9925) on the MNIST dataset, followed by **ResNet** and **AlexNet** with very close performance.
- **Xception** had a noticeably lower performance on MNIST, with an accuracy of 0.9656, which might be due to its design being more suited for more complex tasks.
- **LeNet5** performed well with an accuracy of 0.9900 but lagged slightly behind more modern architectures like VGG16 and ResNet.

## Comparison of CNN Architectures on FMNIST Dataset

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| **AlexNet** | 0.9194   | 0.9194    | 0.9194 | 0.9190   |
| **GoogLeNet** | 0.9176  | 0.9196    | 0.9176 | 0.9180   |
| **VGG16**   | 0.9299   | 0.9301    | 0.9299 | 0.9293   |
| **Xception** | 0.8483  | 0.8501    | 0.8483 | 0.8465   |
| **LeNet5**  | 0.8956   | 0.8954    | 0.8956 | 0.8948   |
| **ResNet**  | 0.8625   | 0.8804    | 0.8625 | 0.8617   |
| **SENet**   | 0.4468   | 0.6149    | 0.4468 | 0.3563   |

### Insights:
- **VGG16** achieved the highest accuracy (0.9299) on the FMNIST dataset, followed closely by **AlexNet** and **GoogLeNet** with similar performance.
- **Xception** performed poorly on FMNIST, with an accuracy of 0.8483, suggesting that this model may not be the best fit for this type of task.
- **ResNet** and **LeNet5** had moderate performance, with **LeNet5** achieving better results than **ResNet**.
- **SENet** had significantly lower performance compared to the other models, possibly due to the architecture being more complex and less suited for this dataset.

## Comparison of CNN Architectures on CIFAR-10 Dataset

| Model        | Accuracy | Precision | Recall  | F1-Score |
|--------------|----------|-----------|---------|----------|
| **AlexNet**  | 0.7556   | 0.7565    | 0.7556  | 0.7550   |
| **VGG16**    | 0.9925   | 0.9926    | 0.9924  | 0.9925   |
| **Xception** | 0.9656   | 0.9660    | 0.9652  | 0.9654   |
| **LeNet5**   | 0.9900   | 0.9899    | 0.9899  | 0.9899   |
| **ResNet**   | 0.9924   | 0.9926    | 0.9922  | 0.9924   |

### Insights:
- **VGG16** achieved the highest accuracy (0.9925) on the CIFAR-10 dataset, followed closely by **ResNet** (0.9924), both of which outperformed the other models significantly.
- **LeNet5** achieved excellent results (accuracy of 0.9900), showcasing strong performance on CIFAR-10.
- **Xception** performed well with an accuracy of 0.9656 but was still behind the top models.
- **AlexNet** achieved the lowest accuracy (0.7556), suggesting it may not be the most efficient model for the CIFAR-10 dataset compared to others.
---

## **Technologies Used**

The projects utilize the following technologies and tools:

- **Programming Language:** Python
- **Libraries and Frameworks:**
  - PyTorch
  - TensorFlow
  - Hugging Face Transformers
  - NumPy, Pandas, Matplotlib
- **Tools:**
  - Jupyter Notebook
  - Streamlit (optional for frontend)
- **Datasets:**
  - Synthetic data for seq2seq
  - Pretrained Hugging Face models (GPT-2, T5, BERT)
  - MNIST, FMNIST, CIFAR-10

---
