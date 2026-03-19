# DeepFLAIR* — C147A Final Project

This work was done independently for C147A, not for my lab. I received the data and help from some lab members, but this project was started and completed on my own for this class.

## Results

| Architecture | Val Loss | Test MSE | Test SSIM |
|---|---|---|---|
| U-Net (ReLU) | 0.0737 | 0.000869 | 0.918 |
| U-Net (Sigmoid) | **0.0617** | 0.000914 | **0.924** |
| Attention U-Net | 0.0750 | 0.000914 | 0.914 |
| Swin-UNETR | 0.0637 | 0.000901 | 0.918 |
| Swin-UNETR (Sigmoid)† | 0.0644 | **0.000789** | 0.922
| UNETR++ (VFA)†† | 0.0801 | 0.000883 | 0.905

† Reached 300 epochs before plateauing. Possibly would have achieved lower loss than Swin-UNETR with ReLU

†† Continued training, improved from results during the paper significantly
