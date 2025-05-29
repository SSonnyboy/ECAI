# 高软报告
## 基于注意力迁移的双学生半监督医学图像分割框架

### Dataset

**Preprocess**: refer to the image pre-processing method in [CoraNet](https://github.com/koncle/CoraNet) and [BCP](https://github.com/DeepMed-Lab-ECNU/BCP) for the Pancreas dataset, Left atrium and ACDC dataset. 
The `dataloaders` folder contains the necessary code to preprocess the Left atrium and ACDC dataset. 
Pancreas pre-processing code can be got at [CoraNet](https://github.com/koncle/CoraNet).

**Dataset split**: The `./Datasets` folder contains the information about the train-test split for all three datasets.

## Usage

Following [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) to prepare the three datasets, and properly setup the path in corresponding config file.

Package: nothing but common ones,

```plaintext
h5py==3.1.0
matplotlib==3.3.4
MedPy==0.4.0
nibabel==3.2.2
numpy==1.19.4
opencv-python==4.4.0.46
pandas==1.1.4
Pillow==8.4.0
PyYAML==5.4.1
scikit-image==0.17.2
scipy==1.5.4
SimpleITK==2.1.1.2
tensorboard==2.4.0
torch==1.10.1
torch-geometric==1.7.0
torchvision==0.8.0a0+45f960c
tqdm==4.54.0=
```



To run, simple check the provided shell files for trianing and testing.

```shell
# setup the config file firstly
# to train on LA
sh ./train_3d_la.sh
# to test on LA
sh ./test_3d_la.sh

```
