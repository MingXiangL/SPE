# End-to-End Weakly Supervised Object Detection with Sparse Proposal Evolution

## Install
```
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
pip install timm
pip install opencv-python
pip install scipy
pip install pycocotools
```

## Data Preperation
We support COCO format. For VOC format annotations, please transfer them to coco format first. 

For example, in experiments on VOC0712 trainval first transfer annotations of VOC2007 trainval and VOC2012 trainval to coco format in a single json file. # note: to evaluate mAP performance, please use the original annotation of voc07 test.

## Models
[VOC0712](https://mailsucaseducn-my.sharepoint.com/:u:/g/personal/liaomingxiang20_mails_ucas_edu_cn/EZqaJfgM6EFGrXarYnc24ysBF5yj-l6iHX4tDgFp7m-eAw?e=5IyFZw): mAP=51.0 

[COCO2017](https://mailsucaseducn-my.sharepoint.com/:u:/g/personal/liaomingxiang20_mails_ucas_edu_cn/Eb-t83JDg-5Diubx-RVlHBgBTnVW9lLn7ghEM8ezUfrDUQ?e=HyABF3): AP=7.9, AP50=19.5, AP75=5.5
## Train & Val

### COCO 2017:
```
python scripts/run_coco17.py
```
### VOC 07+12

```
python scripts/run_voc0712.py
```


