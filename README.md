# Style Transfer with Diffusion and CNN Models

This project provides a unified implementation and comparison of state-of-the-art image style transfer algorithms, with a focus on **Stable Diffusion-based** and **CNN/Transformer-based** methods. The goal is to analyze, reproduce, and evaluate these methods for qualitative and quantitative comparison.

---

## 📁 Project Structure

```
style-transfer/
│
├── Puff-Net/           # Transformer-based efficient style transfer
├── StyleID/            # Diffusion-based training-free style transfer (cross-attention reweighting)
│├── content/            # Content images for style transfer
│├── style/              # Style images for reference
│└── eval/               # Evaluation scripts (ArtFID, LPIPS, CFSD, etc.)
├── Zero-shot/          # Z★: Zero-shot style injection with attention manipulation
├── .gitignore
└── README.md
```

---

## 📚 Included Methods and References

### 🔹 1. **StyleID (Training-free Diffusion Style Transfer)**  
**Paper**: *Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer*  
**Authors**: Jiwoo Chung, Sangeek Hyun, Jae-Pil Heo  
**Conference**: CVPR 2024  
**Code Reference**: [https://github.com/jiwoogit/StyleID](https://github.com/jiwoogit/StyleID)

Key Features:
- Key & Value injection into self-attention layers
- Query preservation and attention temperature scaling
- Initial latent AdaIN for better color stylization

---

### 🔹 2. **Z★ (Zero-shot Style Injection for Diffusion Models)**  
**Paper**: *Zero-shot Style Transfer via Attention Reweighting*  
**Authors**: Z. Deng et al.  
**Conference**: CVPR 2024  
**Code Reference**: [https://github.com/HolmesShuan/Zero-shot-Style-Transfer-via-Attention-Rearrangement](https://github.com/HolmesShuan/Zero-shot-Style-Transfer-via-Attention-Rearrangement)
**Technique**:  
- Based on Stable Diffusion  
- Uses Reweighting of Cross-Attention Maps with Null-text Inversion  
- No training required, pure inference-based style injection

---

### 🔹 3. **Puff-Net (Transformer with Pure Content & Style Feature Fusion)**  
**Paper**: *Puff-Net: Efficient Style Transfer with Pure Content and Style Feature Fusion Network*  
**Authors**: Sizhe Zheng, Pan Gao, Jie Qin  
**Conference**: CVPR 2024  
**Code Reference**: [https://github.com/ZszYmy9/Puff-Net](https://github.com/ZszYmy9/Puff-Net)

Highlights:
- Only uses Transformer encoder for low-latency inference
- Uses INN (Invertible Network) and LT blocks to disentangle content/style
- CAPE positional encoding and AdaIN-based loss

---

## 🔧 Usage
> ✍️ Each subfolder contains its own test/inference script.  
> Refer to individual README files or scripts inside each submodule.

---

## 📊 Evaluation

Metrics used:
- **FID**: Fréchet Inception Distance
- **ArtFID**: Art-optimized FID
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **CFSD**: Content Feature Structural Distance

Evaluation scripts are located in `eval/`.

---

## 📄 License

This project contains original implementations and adapted code from official repositories. See individual submodules for license details.
