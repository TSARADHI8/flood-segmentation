# Flood Extent Segmentation from UAV Imagery

Automatic pixel-level flood detection from drone aerial images using deep learning.
Upload any UAV aerial image and get an instant colour-coded flood map showing water,
flooded buildings, roads, trees, and more.

## Live Demo
🌊 **Web App:** https://huggingface.co/spaces/Saradhi8/flood-segmentation

---

## Project Details

| | |
|---|---|
| **Dataset** | FloodNet Challenge @ EARTHVISION 2021 — Track 1 |
| **Model** | U-Net with pretrained ResNet34 encoder |
| **Overall mIoU** | 59.9% on 398 labelled images |
| **Training** | 30 epochs, batch size 4, Adam optimizer, Tesla T4 GPU |

---

## Results

| Category | Images | mIoU |
|---|---|---|
| Flooded scenes | 51 | 24.7% |
| Non-flooded scenes | 347 | 65.1% |
| **Overall** | **398** | **59.9%** |
| Best single image | 1 | 100.0% |

Training loss dropped from **1.9270 → 0.3428** (82% reduction over 30 epochs).

---

## The 10 Segmentation Classes

| Class ID | Name | Flood-critical |
|---|---|---|
| 0 | Background | |
| 1 | Building Flooded | ✅ |
| 2 | Building Non-Flooded | |
| 3 | Road Flooded | ✅ |
| 4 | Road Non-Flooded | |
| 5 | Water | ✅ |
| 6 | Tree | |
| 7 | Vehicle | |
| 8 | Pool | |
| 9 | Grass | |

---

## How to Run

1. Open `Flood_Segmentation_Complete.ipynb` in Google Colab
2. Enable T4 GPU — Runtime → Change runtime type → T4 GPU
3. Mount Google Drive with FloodNet dataset
4. Run all cells top to bottom
5. Skip the training cell — load the saved model instead

---

## Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Deep learning framework |
| segmentation-models-pytorch | U-Net with pretrained encoders |
| Albumentations | Image augmentation |
| Google Colab T4 GPU | Training environment |
| Streamlit | Web application |
| Hugging Face Spaces | Live deployment |

---

## Dataset

FloodNet Challenge @ EARTHVISION 2021 — Track 1
https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021

Collected after Hurricane Harvey (2017), Texas, USA using DJI Mavic Pro drones.
398 labelled images used for training (51 flooded + 347 non-flooded).

---

## Repository Structure

```
flood-segmentation/
├── Flood_Segmentation_Complete.ipynb  ← main notebook
├── app.py                             ← Streamlit web application
├── requirements.txt                   ← Python dependencies
└── README.md                          ← this file
```
