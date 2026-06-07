# DecodeLabs-Internship-project4
# 🏭 Industrial Training Kit - Image & Text Recognition System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)

## 📌 Project Overview

This project is an **Industrial Training Kit** developed as part of the **DecodeLabs AI Engineer Training Program (Batch 2026)**. It demonstrates a complete recognition pipeline that can either **extract text from images (OCR)** or **detect and locate objects in images** with high confidence.

### 🎯 The Challenge
Over 80% of enterprise data is unstructured (images, scanned docs, video). This project bridges the gap by converting raw visual data into structured, machine-readable intelligence.

---

## ✨ Features

### Dual Recognition Paths

| Path | Technology | Output |
|------|-----------|--------|
| **OCR (Text Recognition)** | EasyOCR / Tesseract | Formatted text strings |
| **Object Detection** | OpenCV DNN + Contour Detection | Bounding boxes with labels |

### ✅ 4 Mandatory Validation Checkpoints

1. **Library Integration** - Seamless use of pytesseract/OpenCV DNN
2. **Pre-Processing Integrity** - Grayscale + Gaussian blur + Adaptive thresholding
3. **Accuracy Benchmarking** - Minimum **80% confidence score** on output
4. **Visual Confirmation** - Clean visual output with bounding boxes/labels

---
## 🧠 How It Works
### OCR Pipeline (Path 1)

**OCR Output**

<img width="1495" height="742" alt="image" src="https://github.com/user-attachments/assets/f778d65c-2d8f-4e3d-863f-7d78e25b5e42" />

Raw Image → Grayscale → Gaussian Blur → Adaptive Thresholding 
    → EasyOCR Inference → Confidence Filtering (≥80%) 
    → Text Output + Visualization
    
### Object Detection Pipeline (Path 2)

**Object Detection Output**

<img width="1499" height="738" alt="image" src="https://github.com/user-attachments/assets/f60862cd-7d6f-4ea6-a92d-2592d822bf37" />

Raw Image → Grayscale → Edge Detection → Contour Detection 
    → Shape Classification → Confidence Filtering (≥80%) 
    → Bounding Boxes + Visualization
## Terminal Output
<img width="465" height="358" alt="image" src="https://github.com/user-attachments/assets/eac4a440-0fcf-4a22-b113-3ce703e635d4" />

