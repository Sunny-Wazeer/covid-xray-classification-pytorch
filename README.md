# COVID-19 Chest X-ray Classification with CNN

This project focuses on building a Convolutional Neural Network (CNN) using PyTorch to classify chest X-ray images into three categories: **Covid-19**, **Normal**, and **Viral Pneumonia**.

## Dataset

The dataset is the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), containing labeled X-ray images for:
- COVID-19
- Normal
- Viral Pneumonia

## Project Steps

1. **Download & Extract Dataset**: 
   - Upload `kaggle.json`
   - Use Kaggle API to download and extract the dataset.

2. **Data Preparation**:
   - Extract image paths and labels
   - Use LabelEncoder to encode string labels to integers
   - Sample 2000 images: 30% COVID, 30% Normal, 40% Pneumonia
   - Split into train, validation, and test sets
   - Define PyTorch `Dataset` and `DataLoader`

3. **Model Definition**:
   - A 3-layer CNN is defined using `torch.nn.Module`

4. **Training**:
   - Adam optimizer, CrossEntropyLoss
   - 20 epochs with accuracy tracking

5. **Evaluation**:
   - Classification report
   - Confusion matrix
   - Accuracy plot (training vs validation)

6. **Prediction System**:
   - Accepts a new image path and displays predicted label with confidence score

## Results

The CNN model achieves solid performance on validation and test data. Accuracy improves consistently with training.

## Usage

Run the notebook step-by-step on Google Colab. Ensure `kaggle.json` is uploaded for dataset download.

## Inference

To predict a new image:
```python
image_path = "/content/image.png"
predicted_label, confidence = detection_system_pytorch(image_path, model, le, image_size=150, device=device)
```

## Files Saved

- `covid_classifier.pth`: Trained model weights
- `label_encoder.pkl`: Label encoder used during training

---

## Author

Developed as a deep learning project using PyTorch and chest X-ray images.
