# UDiCT
The codes for Semi-supervised CT lesion segmentation using uncertainty-based data pairing and swapmix




Implementation:

1. Use UDiCT_env.yaml to configure your environment.

2. Download LIDC-IDRI or COVID-SemiSeg dataset and unzip them.
  The LIDC-IDRI dataset used here is a manually processed version, which can be download at https://pantheon.corp.google.com/storage/browser/hpunet-data/lidc_crops from https://github.com/SimonKohl/probabilistic_unet.

    The Covid-SemiSeg dataset can be download at https://drive.google.com/open?id=1bbKAqUuk7Y1q3xsDSwP07oOXN_GL3SQM from https://github.com/DengPingFan/Inf-Net.

    The Seg-C19 dataset is collected by us privately, please contact us if you need it.
  
3. Add the data folder path to dataload_inf.py and dataload_lidc.py

4. run the .sh file in LIDC and COVIDSemiSeg, the hypaprmeters set in the .sh file are the hypaprmeters used in our paper.
