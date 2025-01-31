# Development of Depth Maps for Outdoor Images

This project focuses on improving depth map generation for outdoor scenes, particularly addressing the challenges of nighttime imaging. We propose and evaluate two distinct approaches to enhance depth estimation quality for low-light conditions.

## 🌟 Features

- Domain adaptation approach using DepthAnything model
- Nighttime image enhancement pipeline
- Support for various enhancement algorithms
- Comparative analysis of depth map quality
- Tested on real-world dataset from Bhopal, India

## 📊 Results

## Daytime vs Nighttime Depth Maps
![Depth Map Comparison](day_busy.png)
*Comparison of depth maps for various scenes: (a) busy daytime


![Depth Map Comparison](day_empty.png)
*Comparison of depth maps for various scenes: (b) empty daytime


![Depth Map Comparison](night_busy.png)
*Comparison of depth maps for various scenes: (c) busy nighttime


![Depth Map Comparison](night_empty.png)
*Comparison of depth maps for various scenes: (d) empty nighttime


![Depth Map Comparison](night_verylowres.png)
*Comparison of depth maps for various scenes: (e) noisy nighttime*


## Domain Adaptation Results
![Domain Adaptation Results](train_day_busy_comparison.png)
![Domain Adaptation Results](train_day_empty_comparison.png)
![Domain Adaptation Results](train_night_busy_comparison.png)
![Domain Adaptation Results](train_night_empty_comparison.png)
![Domain Adaptation Results](train_night_verylowres_comparison.png)
*Results on a busy night-time scene showing actual scene, depth map from DepthAnything v1, difference in depth maps, and depth map from our encoder*

## Image Enhancement Results
![Enhancement Results](Lowres_1.png)
![Enhancement Results](Lowres_2.png)
![Enhancement Results](Lowres.png)
*Results showing enhanced nighttime images along with their corresponding depth maps using various algorithms*

## 🛠️ Approaches

### 1. Domain Adaptation Based
- Uses DepthAnything model architecture
- Fine-tuning and training from scratch experiments
- PatchGAN-based adversarial learning
- Unsupervised approach for feature space alignment

### 2. Nighttime Image Enhancement Based
Implemented various enhancement algorithms including:
- Naturalness Preserved Enhancement (NPE)
- Low-Light Image Enhancement (LIME)
- Fusion-Based Enhancement (FBE)
- Bio-Inspired Multi-Exposure Fusion (BIMEF)
- Robust Retinex Model (RRM)
- And more...

## 📈 Performance

Best results were obtained using:
- NPE (Naturalness Preserved Enhancement)
- FOF (Fractional-Order Fusion)
- CR (Camera Response Model)

## 🚧 Limitations

- Limited dataset size affecting domain adaptation
- Subjective evaluation metrics
- Noise amplification in enhancement
- Loss of texture and detail
- Over-saturation issues in extreme cases

## 🔮 Future Work

- Combining both approaches for robust performance
- Testing with larger datasets
- Exploring extreme weather conditions
- Improving evaluation metrics
- Optimizing hyperparameters

## 📚 References

1. Basheer, O., & Al-Ameen, Z. (2024). Nighttime Image Enhancement: A Review of Topical Concepts. SISTEMASI, 13, 1073.
2. Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision.
3. Vankadari, M., et al. (2020). Unsupervised Monocular Depth Estimation for Night-time Images using Adversarial Domain Feature Adaptation.
4. Yang, L., et al. (2024). Depth Anything V2.
5. Yang, L., et al. (2024). Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data.

## 👥 Authors: Feel free to reach out

- Devashish Tripathi (devashish21@iiserb.ac.in) Github: (https://github.com/Devashish-Tripathi)
- Aditya Kishore (adityak21@iiserb.ac.in) Github: (https://github.com/Adityakishore09)
- Snehal Mahajan (mahajan20@iiserb.ac.in)

## Dataset: Kindly Email us for the access to Dataset

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
