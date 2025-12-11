# PIVFusion
Codes for ***Perceptual Transform Fusion of Infrared and Visible Images***.

[Dingli Hua](), [Qingmao Chen](https://hinmouc.github.io), [Zhiliang Wu](), [Yifan Zuo](), [ Wenying Wen](), [Yuming Fang]().

-[*[Paper]*](https://ieeexplore.ieee.org/document/11078301)  [*[Project]*](https://hinmouc.github.io/PIVFusion/)  

## Abstract
Infrared and visible image fusion aims to generate fused images with rich textures and clear target representations. Existing methods generally assume high-quality input images, thus overlooking issues such as reduced contrast and loss of details in visible images under low-light conditions. The naive enhance-then-fuse strategy cannot perform fuse-oriented image enhancement, which always reaches a sub-optimal result. To address this challenge, we propose a perceptual transform fusion of infrared and visible images, which simultaneously optimizes low-light enhancement and image fusion. Specifically, to improve computational efficiency and optimize key feature representations while suppressing noise interactions caused by lighting variations, we introduce a lightweight adaptive sparse Transformer block (ASTBlock). This model adaptively integrates sparse and dense attention mechanisms to enhance feature representations and employs a feed-forward network to eliminate redundant information, thereby ensuring the quality of image fusion. Subsequently, to retain significant details while reducing the impact of noise introduced by low-light enhancement, we incorporate discrete wavelet transform (DWT) for feature decomposition and fusion, further enhancing the representation capability and feature preservation of fused images. Meanwhile, to tackle the issues of insufficient contrast and hidden details in low-light conditions, we design an illumination perception module and an illumination consistency loss to improve the contrast and clarity of fused images. Experimental results on multiple public benchmark datasets for quality assessment and downstream tasks, e.g., pedestrian detection, demonstrate that our method significantly outperforms the state-of-the-art (SOTA) methods.

## Pipeline

![pipleline](pic//pipeline.png)



## Demo
### 🚀 Train
Modify the variable 'dataset' on  ``train.py``, and run
```
python train.py
```

### 🧪 Test

**1. Pretrained models**

Pretrained models are available in [checkpoint]().

**2. Test cases**

The 'test_cases' folder contains seven examples that appear in the main paper. Running 

```
python test.py
```

will fuse these cases, and the fusion results will be saved in the ``./test_result`` folder.



## 📝Citation

If this work is helpful to you, please cite it as:
```
@ARTICLE{2025pivfusion,
  title={Perceptual Transform Fusion of Infrared and Visible Images},
  author={Hua, Dingli and Chen, Qingmao and Wu, Zhiliang 
  		  and Zuo, Yifan and Wen, Wenying and Fang, Yuming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
}
```

