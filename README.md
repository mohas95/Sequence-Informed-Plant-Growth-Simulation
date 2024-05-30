# Sequence Informed Generative Plant Growth Simulation

**Corresponding Paper:** [ArXiv](https://arxiv.org/abs/2405.14796)


## Supplemental Figures

### Fig. S1

![](/figures/S1_outputs.gif)

### Fig. S2

![](/figures/S2_beta.gif)

### Fig. S3
#### LSTM Encoder
| Layer Type         | Output Shape         | Kernel Size / Stride / Padding | Activation Function |
|--------------------|----------------------|--------------------------------|---------------------|
| Input              | (None, sequence_length, feature_size) |                                |                     |
| LSTM               | (None, sequence_length, hidden_size)  |                                |                     |
| Fully Connected    | (None, output_size)                    |                                | Tanh                |


#### CVAE_Generator Architecture
| Layer Type         | Output Shape            | Kernel Size / Stride / Padding | Activation Function |
|--------------------|-------------------------|--------------------------------|---------------------|
| Input              | (None, input_channels, height, width) |                                |                     |
| Conv2d             | (None, 64, height/2, width/2)          | 4x4 / 2 / 1                    | ReLU                |
| Conv2d             | (None, 128, height/4, width/4)         | 4x4 / 2 / 1                    | ReLU                |
| Conv2d             | (None, 256, height/8, width/8)         | 4x4 / 2 / 1                    | ReLU                |
| Conv2d             | (None, 512, height/16, width/16)       | 4x4 / 2 / 1                    | ReLU                |
| Linear (mu)        | (None, latent_size)                    |                                |                     |
| Linear (logvar)    | (None, latent_size)                    |                                |                     |
| Linear (z)         | (None, hidden_size)                    |                                |                     |
| ConvTranspose2d    | (None, 256, height/8, width/8)         | 4x4 / 2 / 1                    | ReLU                |
| ConvTranspose2d    | (None, 128, height/4, width/4)         | 4x4 / 2 / 1                    | ReLU                |
| ConvTranspose2d    | (None, 64, height/2, width/2)          | 4x4 / 2 / 1                    | ReLU                |
| ConvTranspose2d    | (None, 3, height, width)               | 4x4 / 2 / 1                    | Sigmoid             |

#### Discriminator Architecture
| Layer Type         | Output Shape            | Kernel Size / Stride / Padding | Activation Function |
|--------------------|-------------------------|--------------------------------|---------------------|
| Input              | (None, input_channels, height, width) |                                |                     |
| Conv2d             | (None, 64, height/2, width/2)          | 4x4 / 2 / 1                    | ReLU                |
| Conv2d             | (None, 128, height/4, width/4)         | 4x4 / 2 / 1                    | ReLU                |
| Conv2d             | (None, 256, height/8, width/8)         | 4x4 / 2 / 1                    | ReLU                |
| Conv2d             | (None, 512, height/16, width/16)       | 4x4 / 2 / 1                    | ReLU                |
| Fully Connected    | (None, 512)                             |                                | ReLU                |
| Fully Connected    | (None, 1)                               |                                | Sigmoid             |


## Required Python Libaries
``` bash
# Jupyter-lab
pip install jupyterlab
# pytorch
pip install torch torchvision torchaudio
# pandas
pip install pandas
# matplotlib
pip install matplotlib
# scipy
pip install scipy
# opencv
pip install opencv-python
# Sci-kit learn
pip install scikit-learn
# tqdm
pip install tqdm
# ipywidgets
pip install ipywidgets

```
## Cite this work
```bibtex
@misc{debbagh2024generative,
      title={Generative Plant Growth Simulation from Sequence-Informed Environmental Conditions}, 
      author={Mohamed Debbagh and Yixue Liu and Zhouzhou Zheng and Xintong Jiang and Shangpeng Sun and Mark Lefsrud},
      year={2024},
      eprint={2405.14796},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```