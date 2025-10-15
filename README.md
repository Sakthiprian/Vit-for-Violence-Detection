# Violence Detection using Vision Transformers

Final year deep learning project for detecting violent activities in surveillance videos using **Vision Transformers (ViT)** and **TimeSformer** architectures.  
Developed as part of the **ECE Department Project, IIITDM Chennai** by *Sakthiprian S (EC21B1075)* under the supervision of **Dr. Umarani Jayaraman** and **Dr.Pal Uttam Mrinal**.

---

## 🧠 Project Overview

With the growing need for public safety, automated surveillance systems are crucial for identifying violent activities in real-time.  
Traditional CNN-based architectures struggle with modeling long-range dependencies across frames.  
This project leverages **Vision Transformers (ViTs)** — models that apply self-attention to both spatial and temporal features — to detect violence in videos more effectively.

---

## 🎯 Objectives

- Develop a **ViT-based model** for video violence detection.  
- Use **self-attention mechanisms** to improve classification accuracy.  
- Fine-tune on open-source datasets to benchmark performance.  
- Compare two approaches:
  1. **Optical Flow + ViT**
  2. **Spatiotemporal Attention (TimeSformer)**

---

## 📂 Dataset

**Dataset used:** [RWF-2000](https://github.com/fjchange/RWF2000-Video-Database)  
- 2000 labeled surveillance videos (1000 Fight, 1000 Non-Fight)  
- Each video contains ~150 frames  
- Frames are split into **16-frame non-overlapping clips**  

**Final structure:**
```
clips_16/
├── train/
│   ├── Fight/
│   └── NotFight/
└── val/
    ├── Fight/
    └── NotFight/
```

---

## ⚙️ Methodology

### 1. Optical Flow + ViT

- **Optical flow estimation:** Motion between consecutive frames.  
- **Momentum-based accumulation:**
  acc_flow = α × curr_flow + (1 − α) × prev_acc_flow, α = 0.7  
- Input to model: `16th RGB frame + accumulated flow`
- **Model:** ViT-Base (patch size 16, image size 224)
- **Optimizer:** AdamW (LR: 1e-4)
- **Epochs:** 20 | **Batch size:** 128

### 2. TimeSformer (Spatiotemporal Attention)

- Extends ViT with **temporal attention** to model motion across frames.  
- Input: 16 RGB frames per clip.  
- **Model:** TimeSformer-Base  
- **Optimizer:** AdamW (LR: 1e-4)
- **Epochs:** 14 | **Batch size:** 14

---

## 🧩 Model Architecture

- **Framework:** PyTorch  
- **Backbone:** Vision Transformer (ViT) / TimeSformer  
- **Loss function:** CrossEntropyLoss  
- **Scheduler:** ReduceLROnPlateau  
- **Hardware:** Google Colab L4 GPU

---

## 📊 Results

| Model | Validation Accuracy |
|--------|----------------------|
| ViT + Optical Flow | **97.2%** |
| TimeSformer | **95.2%** |

Both models effectively captured spatial-temporal relationships in the data, achieving high accuracy on validation sets.

---

## 🔍 Future Scope

- Train with **larger datasets** and **longer clip lengths**.  
- Optimize for **real-time inference** on edge devices.  
- Extend to **multi-class anomaly detection** in surveillance systems.

---

## 📚 References

- Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, arXiv:2010.11929.  
- Singh et al. *Video Vision Transformers for Violence Detection*, arXiv:2209.03561.  
- Rahman et al. *Lightweight Vision Transformers for Real-Time Violence Detection in Indoor Surveillance*, PRL, 2023.  
- Bertasius et al. *Is Space-Time Attention All You Need for Video Understanding?*, ICML 2021.  

---

## 👨‍💻 Author

**Sakthiprian S**  
Department of Electronics & Communication Engineering  
**IIITDM Chennai**  
📧 *sakthiprians@gmail.com*
