# UDiCT
The codes for Semi-supervised CT Lesion Segmentation Using Uncertainty-based Data Pairing and Swapmix




Implementation:

1. Use UDiCT_env.yaml to configure your environment.

2. Download LIDC-IDRI or COVID-SemiSeg dataset and unzip them.
  The LIDC-IDRI dataset used here is a manually processed version, which can be download at https://pantheon.corp.google.com/storage/browser/hpunet-data/lidc_crops from https://github.com/SimonKohl/probabilistic_unet.

    The Covid-SemiSeg dataset can be download at https://drive.google.com/open?id=1bbKAqUuk7Y1q3xsDSwP07oOXN_GL3SQM from https://github.com/DengPingFan/Inf-Net.

    The Seg-C19 dataset is collected by us privately, please contact us if you need it.
  
3. Add the data folder path to dataload_inf.py, dataload_segc19.py and dataload_lidc.py

4. run the .sh file in LIDC, SegC19 and COVIDSemiSeg to train your model.

Please consider citing this project in your publications if it helps your research.
```bibtex
@article{qiao2022semi,
  title={Semi-Supervised CT Lesion Segmentation Using Uncertainty-Based Data Pairing and SwapMix},
  author={Qiao, Pengchong and Li, Han and Song, Guoli and Han, Hu and Gao, Zhiqiang and Tian, Yonghong and Liang, Yongsheng and Li, Xi and Zhou, S Kevin and Chen, Jie},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}
```
