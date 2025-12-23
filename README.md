# PCRNet: Progressive Correlation Refinement for Few-Shot Hyperspectral Image Classification

The source code for our new work on Few-Shot Hyperspectral Image Classification tasks. The details will be reported after the publication.

## Datasets

```
├── Patch9_TRIAN_META_DATA_imdb.pickle
├── test_datasets
│   ├── PaviaU_data.mat
│   ├── PaviaU_gt.mat
│   ├── Indian_pines_corrected.mat
│   ├── Indian_pines_gt.mat
│   ├── Salinas_corrected.mat
│   ├── Salinas_gt.mat
│   ├── WHU_Hi_HanChuan_gt.mat
│   ├── WHU_Hi_HanChuan.mat

```
1) Run "trainMetaDataProcess.py" to generate the meta-training data "Patch5_TRIAN_META_DATA_imdb_ocbs.pickle" and "Patch17_TRIAN_META_DATA_imdb_ocbs.pickle". And you can choose to download the meta-training data through Quark Netdisk (link:https://pan.quark.cn/s/47eeb0e98b00?pwd=7dwc)
2) Run "python DFHE.py".
