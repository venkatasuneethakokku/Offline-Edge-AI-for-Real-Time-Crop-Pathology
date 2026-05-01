# Crop Disease Detection System - Complete Project Documentation

## Project Overview
This is an AI-powered crop disease detection system that uses deep learning to identify diseases in crop images (Corn, Potato, Rice, Wheat, and Sugarcane) and provides treatment recommendations.

---

## 1. TRAINING ALGORITHMS & METHODOLOGIES

### 1.1 Base Architecture: MobileNetV2
- **Why MobileNetV2?**
  - Lightweight (optimized for mobile/edge devices)
  - Fast inference with lower latency
  - Pre-trained on ImageNet weights (transfer learning)
  - Reduced model size compared to traditional CNNs

### 1.2 Transfer Learning Approach
- **Step 1: Initial Training (15 epochs)**
  - Base MobileNetV2 backbone is **frozen** (not trainable)
  - Only the top dense layers are trained
  - Learning rate: **0.001** (Adam optimizer)
  - Fast training, prevents overfitting

- **Step 2: Fine-Tuning (10 epochs)**
  - Unfreeze last 20 layers of MobileNetV2
  - Further layers remain frozen
  - Learning rate: **0.00001** (reduced 100x)
  - Gradually adapt pre-trained weights to crop disease data

### 1.3 Additional Techniques

**Data Augmentation:**
```
- Random horizontal flips
- Random rotations (10 degrees max)
- Random zoom (10% variation)
Purpose: Increases training data diversity, prevents overfitting
```

**Learning Rate Reduction:**
- ReduceLROnPlateau callback reduces learning rate automatically
- Triggers if validation loss plateaus for 2 epochs
- Factor: 0.2 (multiply by 0.2)
- Minimum LR: 1e-7

**Early Stopping:**
- Monitors validation loss
- Patience: 5 epochs (stops if no improvement)
- Saves best model weights automatically

---

## 2. LIBRARIES USED & WHY

### TensorFlow (2.14.0)
```
What: Google's deep learning framework
Why used:
- Pre-trained MobileNetV2 model included
- Easy model building with Keras API
- GPU acceleration for training
- TFLite conversion for mobile deployment
```

### Keras (part of TensorFlow)
```
What: High-level neural network API
Why used:
- Simple Sequential/Model API
- Built-in callbacks (EarlyStopping, ReduceLROnPlateau)
- Data augmentation layers
- Model evaluation tools
```

### Scikit-learn
```
What: Machine learning utilities library
Why used:
- Classification reports (precision, recall, F1-score)
- Confusion matrix generation
- Metric calculations
```

### NumPy
```
What: Numerical computing library
Why used:
- Array operations for image data
- Quantization/normalization of model outputs
- Matrix computations
```

### Pillow (PIL)
```
What: Image processing library
Why used:
- Load images from bytes
- Resize images to fixed dimensions (224x224)
- Convert images to RGB format
```

### Matplotlib & Seaborn
```
What: Data visualization libraries
Why used:
- Plot training/validation loss curves
- Plot training/validation accuracy curves
- Visualize confusion matrices with heatmaps
- Generate reports for analysis
```

### FastAPI
```
What: Modern web framework
Why used:
- Build REST API for predictions
- Automatic documentation (/docs endpoint)
- Async support for concurrent requests
- CORS middleware for frontend integration
```

### Uvicorn
```
What: ASGI server
Why used:
- Run FastAPI application
- Handle HTTP requests
- Async support for multiple simultaneous predictions
```

---

## 3. MODEL ARCHITECTURE DETAILS

### Input
- **Shape**: 224 × 224 × 3 (224x224 RGB images)
- **Preprocessing**: MobileNetV2 standard preprocessing

### Network Layers
```
1. Data Augmentation Layer
   ├─ Random Flip
   ├─ Random Rotation (±10°)
   └─ Random Zoom (±10%)

2. Preprocessing Layer
   └─ MobileNetV2 preprocess_input

3. MobileNetV2 Base Model (transfer learning)
   └─ Pre-trained on ImageNet
   └─ Parameterized bottleneck layers
   └─ 3.5M parameters reduction vs full CNN

4. Global Average Pooling
   └─ Reduces spatial dimensions

5. Dropout Layer
   └─ Rate: 0.3 (prevents overfitting)

6. Dense Output Layer
   ├─ Activation: Softmax
   └─ Units: 17 (number of disease classes)
```

### Output
- **17 Disease Classes Detected:**
  1. Corn - Common Rust
  2. Corn - Gray Leaf Spot
  3. Corn - Healthy
  4. Corn - Northern Leaf Blight
  5. Potato - Early Blight
  6. Potato - Healthy
  7. Potato - Late Blight
  8. Rice - Brown Spot
  9. Rice - Healthy
  10. Rice - Leaf Blast
  11. Rice - Neck Blast
  12. Wheat - Brown Rust
  13. Wheat - Healthy
  14. Wheat - Yellow Rust
  15. Sugarcane - Bacterial Blight
  16. Sugarcane - Healthy
  17. Sugarcane - Red Rot

---

## 4. MODEL ACCURACY & PERFORMANCE

### Overall Accuracy: **92%**

### Per-Class Performance:

| Disease Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Corn - Common Rust | 1.00 | 1.00 | 1.00 | 235 |
| Corn - Gray Leaf Spot | 0.92 | 0.70 | 0.79 | 99 |
| Corn - Healthy | 1.00 | 1.00 | 1.00 | 220 |
| Corn - Northern Leaf Blight | 0.86 | 0.98 | 0.92 | 201 |
| Potato - Early Blight | 0.99 | 0.98 | 0.99 | 186 |
| Potato - Healthy | 0.93 | 0.96 | 0.95 | 28 |
| Potato - Late Blight | 0.98 | 0.98 | 0.98 | 198 |
| Rice - Brown Spot | 0.68 | 0.61 | 0.64 | 122 |
| Rice - Healthy | 0.78 | 0.91 | 0.84 | 316 |
| Rice - Leaf Blast | 0.69 | 0.55 | 0.61 | 186 |
| Rice - Neck Blast | 1.00 | 1.00 | 1.00 | 195 |
| Wheat - Brown Rust | 0.99 | 0.96 | 0.98 | 184 |
| Wheat - Healthy | 0.98 | 0.99 | 0.99 | 242 |
| Wheat - Yellow Rust | 0.99 | 0.99 | 0.99 | 188 |
| Sugarcane - Bacterial Blight | 0.95 | 0.78 | 0.86 | 23 |
| Sugarcane - Healthy | 0.89 | 1.00 | 0.94 | 16 |
| Sugarcane - Red Rot | 0.85 | 0.92 | 0.88 | 25 |

### Key Metrics:
- **Macro Average F1-Score**: 0.90 (average across all classes)
- **Weighted Average F1-Score**: 0.91 (weighted by support)
- **Total Test Images**: 2,664

### Top Performing Classes (Perfect Accuracy):
- ✅ Corn - Common Rust (1.00)
- ✅ Corn - Healthy (1.00)
- ✅ Rice - Neck Blast (1.00)

### Classes Needing Improvement:
- Rice - Leaf Blast (0.61 F1-score)
- Rice - Brown Spot (0.64 F1-score)

---

## 5. TRAINING PROCESS

### Hyperparameters:
```
Initial Training:
  - Epochs: 15
  - Batch Size: 16
  - Learning Rate: 0.001 (Adam)
  - Loss Function: SparseCategoricalCrossentropy
  - Metric: Accuracy

Fine-Tuning:
  - Epochs: 10
  - Batch Size: 16
  - Learning Rate: 0.00001
  - Last 20 MobileNetV2 layers unfrozen
  - Rest of backbone remains frozen

Total Training: 25 epochs
```

### Data Split:
```
Training Set: 80% of dataset
Validation Set: 20% of dataset
Random seed: 42 (reproducibility)
```

### Callbacks Used:
1. **EarlyStopping**
   - Monitor: validation loss
   - Patience: 5 epochs
   - Restores best weights

2. **ReduceLROnPlateau**
   - Monitor: validation loss
   - Factor: 0.2
   - Patience: 2 epochs
   - Min LR: 1e-7

3. **ModelCheckpoint**
   - Saves best model as `best_model.h5`
   - Monitors validation loss

---

## 6. DATASET INFORMATION

### Dataset Structure:
```
Crop___Disease/
├── Corn___Common_Rust/ (235 images)
├── Corn___Gray_Leaf_Spot/ (99 images)
├── Corn___Healthy/ (220 images)
├── Corn___Northern_Leaf_Blight/ (201 images)
├── Potato___Early_Blight/ (186 images)
├── Potato___Healthy/ (28 images)
├── Potato___Late_Blight/ (198 images)
├── Rice___Brown_Spot/ (122 images)
├── Rice___Healthy/ (316 images)
├── Rice___Leaf_Blast/ (186 images)
├── Rice___Neck_Blast/ (195 images)
├── Wheat___Brown_Rust/ (184 images)
├── Wheat___Healthy/ (242 images)
├── Wheat___Yellow_Rust/ (188 images)
├── sugarcane__Bacterial Blight/ (23 images)
├── sugarcane__Healthy/ (16 images)
└── sugarcane__Red Rot/ (25 images)
```

### Total Images: ~2,664 (for evaluation)
### Total Classes: 17
### Image Format: JPG, PNG, BMP, TIFF, WebP
### Standard Image Size: 224 × 224 pixels

---

## 7. DATA PREPROCESSING

### Input Image Preprocessing:
```python
1. Load image from bytes (supports multiple formats)
2. Convert to RGB (if grayscale or RGBA)
3. Resize to 224 × 224
4. Convert to numpy array (float32)
5. Add batch dimension: (1, 224, 224, 3)
```

### During Training:
```python
1. Data augmentation (random flip, rotation, zoom)
2. MobileNetV2 preprocessing_input normalization
3. Batch processing (16 images per batch)
```

### During Inference:
```python
1. Image preprocessing (same as above)
2. Input tensor quantization (INT8 for TFLite)
3. Model prediction
4. Output dequantization
5. Probability normalization (softmax)
```

---

## 8. MODEL DEPLOYMENT & OPTIMIZATION

### Model Files:
```
backend/ml/models/
├── best_model.h5          (117 MB - Keras format)
├── final_model.h5         (117 MB - Complete trained model)
├── crop_model.tflite      (29 MB - Optimized for deployment)
└── class_names.json       (17 class labels)
```

### Optimization Technique: TFLite INT8 Quantization
```
What: Quantization
Why: Reduces model size from 117MB to 29MB (~75% reduction)
How: Converts float32 weights to int8
Benefit: Faster inference, less memory, minimal accuracy loss
```

**Quantization Parameters:**
- Representative dataset: 100 random training images
- Input type: INT8
- Output type: INT8
- Optimization algorithm: DEFAULT

### Model Inference:
```
Best Model Path: backend/ml/models/crop_model.tflite
Interpreter: TensorFlow Lite
Input shape: (1, 224, 224, 3) INT8
Output shape: (1, 17) INT8 (17 disease probabilities)
```

---

## 9. PROJECT ARCHITECTURE

### Backend Structure:
```
backend/
├── app/                    (FastAPI application)
│   ├── main.py            (Flask app, routes, lifespan)
│   ├── routes/            (API endpoints)
│   │   └── predict.py     (POST /predict endpoint)
│   ├── services/          (Business logic)
│   │   ├── inference_service.py   (Model prediction)
│   │   └── preprocessing_service.py (Image preprocessing)
│   ├── schemas/           (Response data models)
│   │   └── response_schema.py
│   └── data/
│       └── disease_info.json (Disease descriptions, symptoms, treatment)
│
└── ml/                    (Training pipeline)
    ├── training/
    │   ├── model_builder.py
    │   ├── trainer.py
    │   ├── data_loader.py
    │   ├── evaluator.py
    │   └── convert_tflite.py
    ├── models/           (Trained models)
    ├── reports/          (Evaluation results)
    └── run_training.py   (Main training script)
```

### Frontend Structure:
```
frontend/
└── index.html            (Single-page HTML interface)
    ├── Image upload input
    ├── Predict button
    ├── Image preview
    └── Results display:
        - Disease name
        - Confidence %
        - Description
        - Symptoms
        - Treatment
        - Prevention
```

---

## 10. API ENDPOINT

### POST /predict
```
Request:
  - Content-Type: multipart/form-data
  - File: Image file (jpeg, png, webp, bmp, tiff)
  - Max size: 10 MB

Response (200 OK):
{
  "disease_name": "Corn - Common Rust",
  "confidence": 0.9876,
  "description": "A fungal disease...",
  "symptoms": ["Small pustules...", "Reddish-brown..."],
  "treatment": ["Apply fungicides...", "Monitor field..."],
  "prevention": ["Use resistant hybrids...", ...]
}

Error Response (400/422/500):
{
  "detail": "Error message"
}
```

---

## 11. RUNNING THE APPLICATION

### Installation:
```bash
pip install -r requirements.txt
```

### Start Backend:
```bash
cd C:\Users\91918\OneDrive\Desktop\crop
python backend\app\main.py
```

### Access Frontend:
```
http://127.0.0.1:19900
```

### Optional - Retrain Model:
```bash
python -m backend.ml.run_training --dataset-root Crop___Disease
```

---

## 12. KEY FEATURES

✅ **17 Disease Detection** - Covers major crop diseases  
✅ **92% Accuracy** - Validated on 2,664 test images  
✅ **Real-time Inference** - Fast predictions (<1 second)  
✅ **Disease Metadata** - Symptoms, treatment, prevention info  
✅ **Lightweight Model** - 29 MB TFLite format  
✅ **Web Interface** - User-friendly HTML frontend  
✅ **REST API** - Integrates with any application  
✅ **Transfer Learning** - Pre-trained MobileNetV2 backbone  
✅ **Data Augmentation** - Prevents overfitting  
✅ **Model Optimization** - INT8 quantization for fast inference  

---

## 13. TECHNICAL SUMMARY FOR PRESENTATION

### What Problem Does It Solve?
"Early detection of crop diseases helps farmers take preventive action, reduce crop loss, and improve yield."

### How Does It Work?
1. Farmer uploads crop leaf image
2. Model extracts features using MobileNetV2
3. Deep neural network classifies into 17 disease classes
4. System returns disease info + treatment recommendations

### Why MobileNetV2?
- Optimized for mobile/edge deployment
- Pre-trained weights (transfer learning)
- 75% smaller after quantization
- Fast inference

### Model Performance:
- Overall accuracy: **92%**
- Best: 100% on Common Rust, Healthy Corn, Neck Blast
- Type: Convolutional Neural Network with transfer learning

### Training Technique:
- **Phase 1**: Train top layers (frozen backbone) - 15 epochs
- **Phase 2**: Fine-tune backbone layers - 10 epochs
- **Augmentation**: Flip, rotate, zoom for robustness
- **Optimization**: INT8 quantization for deployment

---

## 14. DEPLOYMENT & SHARING

### Requirements File:
- fastapi==0.104.1
- uvicorn==0.24.0
- tensorflow==2.14.0
- pillow==10.1.0
- numpy==1.24.3

### To Share with Friend:
1. Zip without `Crop___Disease/` folder (model already trained)
2. Include: backend/, frontend/, requirements.txt, README.md
3. Friend installs: `pip install -r requirements.txt`
4. Friend runs: `python backend\app\main.py`
5. Access: `http://127.0.0.1:19900`

---

## 15. TECHNOLOGY STACK SUMMARY

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Deep Learning | TensorFlow | 2.14.0 | Model building & training |
| Base Model | MobileNetV2 | Pre-trained | Feature extraction |
| Web Framework | FastAPI | 0.104.1 | REST API |
| Server | Uvicorn | 0.24.0 | ASGI server |
| Image Processing | Pillow | 10.1.0 | Image handling |
| Numeric Ops | NumPy | 1.24.3 | Array operations |
| Metrics | Scikit-learn | Latest | Evaluation metrics |
| Visualization | Matplotlib/Seaborn | Latest | Training plots |

---

## 16. ADVANTAGES OF YOUR APPROACH

1. **Transfer Learning** - Leverages ImageNet knowledge, needs less data
2. **Data Augmentation** - Improves generalization with limited data
3. **Quantization** - 75% size reduction without major accuracy loss
4. **Real-world Deployment** - TFLite format works on edge devices
5. **Comprehensive System** - Frontend + Backend + API fully integrated
6. **Disease Metadata** - Not just predictions, but actionable recommendations
7. **Thread-safe Inference** - Can handle concurrent requests
8. **Early Stopping** - Prevents overfitting automatically
9. **Learning Rate Scheduling** - Adapts learning rate dynamically
10. **Mixed Precision** - INT8 quantization balances speed & accuracy

---
