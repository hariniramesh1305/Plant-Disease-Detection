# 🌱 AI-Based Plant Disease Detection

## 📌 Project Overview

Plant Disease Detection is an AI-based web application that identifies plant diseases from uploaded leaf images.

The application uses a pretrained Vision Transformer (ViT) deep learning model to analyze leaf images, predict the disease, display the confidence score, and provide basic recommended solutions.

## 🎯 Objectives

- Detect plant diseases using leaf images.
- Provide quick and simple disease predictions.
- Display prediction confidence.
- Suggest possible solutions and disease management steps.
- Help users identify plant health problems.

## 🛠️ Technologies Used

- **Python** – Main programming language
- **Gradio** – Web application interface
- **PyTorch** – Deep learning framework
- **Hugging Face Transformers** – Pretrained AI model integration
- **Vision Transformer (ViT)** – Image classification model
- **Pillow (PIL)** – Image processing
- **Google Colab** – Cloud execution environment
- **Cloudflare Tunnel** – Temporary public access during demonstration

## ⚙️ How It Works

1. Open the Plant Disease Detection application.
2. Upload a plant leaf image.
3. Click the Analyze Image button.
4. The image is processed by the AI model.
5. The model predicts the plant disease.
6. The application displays the predicted disease and confidence score.
7. Recommended solutions are displayed based on the prediction.

## 🧠 AI Model

The application uses a pretrained Vision Transformer (ViT) image classification model from Hugging Face:

`kimcomehome/plantvillage-vit-leaf-disease`

The model analyzes the uploaded image and predicts the most likely class.

## 💻 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/hariniramesh1305/Plant-Disease-Detection.git
```

### 2. Open the project folder

```bash
cd Plant-Disease-Detection
```

### 3. Install the required packages

```bash
pip install gradio torch transformers safetensors pillow
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the local application

Open the local URL displayed in the terminal, usually:

`http://127.0.0.1:7860`

The model may download automatically the first time you run the application.

## 🚀 Future Enhancements

- Support for more plant species and diseases.
- Improve prediction accuracy through model fine-tuning.
- Add disease severity detection.
- Provide multilingual recommendations.
- Develop a mobile application.

## 👩‍💻 Developer

**Harini Ramesh**

AI & Data Science Engineering Student

## 📄 Note

This project is developed for educational purposes. Predictions and recommended solutions are intended as preliminary guidance and should not replace expert agricultural advice.
