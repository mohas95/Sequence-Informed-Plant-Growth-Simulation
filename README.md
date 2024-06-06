# Sequence Informed Generative Plant Growth Simulation

**Corresponding Paper:** [ArXiv](https://arxiv.org/abs/2405.14796)


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

## Supplemental Figures

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
ommit line 2 `source venv/bin/activate` if you are not using a virtual environment

Jupyer lab will open, now open [Sequence-Informed-Plant-Growth-Simulation/Run-Plant-Simulation.ipynb](/Run-Plant-Simulation.ipynb) in the browser interface and follow notebook instructions.