# 🚗 CNN Vehicle Image Classifier

A deep learning web app built with **TensorFlow** and **Streamlit** that classifies vehicle images into 5 categories.

## Classes
- Bus
- Car
- Motorcycle
- Truck
- Van

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
[Vehicles Image Dataset](https://www.kaggle.com/datasets/mmohaiminulislam/vehicles-image-dataset) from Kaggle

## Model
Custom CNN with 4 Conv blocks, BatchNormalization, Dropout, and GlobalAveragePooling.
Trained for 40 epochs on 128x128 images.
