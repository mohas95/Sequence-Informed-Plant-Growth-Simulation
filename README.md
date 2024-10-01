# Sequence Informed Generative Plant Growth Simulation

**Corresponding Paper:** [Springer](https://doi.org/10.1007/978-3-031-71602-7_26), [ArXiv](https://arxiv.org/abs/2405.14796)


**Abstract:** A plant growth simulation can be characterized as a reconstructed visual representation of a plant or plant system. The phenotypic characteristics and plant structures are controlled by the scene environment and other contextual attributes. Considering the temporal dependencies and compounding effects of various factors on growth trajectories, we formulate a probabilistic approach to the simulation task by solving a frame synthesis and pattern recognition problem. We introduce a Sequence-Informed Plant Growth Simulation framework (SI-PGS) that employs a conditional generative model to implicitly learn a distribution of possible plant representations within a dynamic scene from a fusion of low dimensional temporal sensor and context data. Methods such as controlled latent sampling and recurrent output connections are used to improve coherence in plant structures between frames of predictions. In this work, we demonstrate that SI-PGS is able to capture temporal dependencies and continuously generate realistic frames of a plant scene.

## Cite this work
```bibtex
debbagh2024generative
@InProceedings{10.1007/978-3-031-71602-7_26,
author="Debbagh, Mohamed
and Liu, Yixue
and Zheng, Zhouzhou
and Jiang, Xintong
and Sun, Shangpeng
and Lefsrud, Mark",
title="Generative Plant Growth Simulation from Sequence-Informed Environmental Conditions",
booktitle="Artificial Neural Networks in Pattern Recognition",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="308--319",
isbn="978-3-031-71602-7"
}
```

## Supplemental Figures

**Please allow a few second for figures to load*
### Fig. S1

![](/figures/S1_outputs.gif)

### Fig. S2

![](/figures/S2_beta.gif)

## Installation

### Clone repository
``` bash
git clone https://github.com/mohas95/Sequence-Informed-Plant-Growth-Simulation.git
cd Sequence-Informed-Plant-Growth-Simulation
```

### Install the required Python Libraries
#### create a virtual environment(optional)
install [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) dependencies
```bash 
sudo apt update
sudo apt install python3-pip
sudo pip3 install virtualenv 
```
create virtual python environment and activate

```bash
virtualenv venv
source venv/bin/activate

```
#### Install libraries

```bash
pip install -r pip_requirements.txt
```
or alternatively install each library individually

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

## Usage
### Plant-Simulation Jupyter Notebook
A user-friendly jupyter notebook lets you run the model on custom user inputs and environmental data. An example input data is given by [example_data.csv](/example_data.csv). Please ensure that custom CSV files are in the same format with the exact same headers. To run the notebook:

``` bash
cd Sequence-Informed-Plant-Growth-Simulation
source venv/bin/activate
jupyter-lab
```
omit line 2 `source venv/bin/activate` if you are not using a virtual environment

Jupyter lab will open, now open [Sequence-Informed-Plant-Growth-Simulation/Run-Plant-Simulation.ipynb](/Run-Plant-Simulation.ipynb) in the browser interface and follow notebook instructions.