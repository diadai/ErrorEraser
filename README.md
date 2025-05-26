# ErrorEraser
 The ErrorEraser plugin code is designed to facilitate the forgetting of erroneous knowledge in continual learning (CL) methods. 

 ![image](https://github.com/user-attachments/assets/6d770e83-5dd6-47b5-8427-1d42f051dd9b)


## 📦 Environment Setup

To ensure reproducibility, we provide the exact environment specification used during development.

### ✅ Required Dependencies

- Python 3.8+
- PyTorch 2.4.1 with CUDA 11.8
- NumPy 1.24+
- Pandas 2.0+
- Scikit-learn 1.3+
- nflows (customized ResNet layers used)
- tqdm
- Matplotlib / Seaborn (optional for visualization)

### 🔧 Installation Instructions

We strongly recommend using **conda** for environment management.

#### Step 1: Create and activate environment

```bash
conda create -n cl-nflow python=3.8
conda activate cl-nflow

#### Step 2: Install base dependencies

```bash
conda install pytorch=2.4.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy pandas scikit-learn tqdm matplotlib seaborn -c conda-forge
```

### ✏️ Reference
The normalizaing flow code refers to [AF-FCL](https://github.com/zaocan666/AF-FCL).
Many thanks to the original authors!
