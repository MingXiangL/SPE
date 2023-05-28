import numpy 
import os
import sys
import random

def run_train(device='0', layer_to_det=24):
    scriptname = os.path.basename(sys.argv[0])[:-3]
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    path_to_coco = "data/coco"
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch --master_port {port}\
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main_coco.py \
                    --epochs 14 \
                    --lr_drop 11 \
                    --dataset_file coco \
                    --fixed_size \
                    --lr_backbone 1e-5 \
                    --lr_cls_head 1e-4 \
                    --batch_size 1 \
                    --enc_layers 3 \
                    --layer_to_det {layer_to_det} \
                    --focal_gamma 0.5 \
                    --backbone TSCAM_cait_XXS36_Two_Branch \
                    --max_size 512 \
                    --num_queries 300 \
                    --weight_decay 5e-2 \
                    --backbone_drop_rate 0.07 \
                    --drop_path_rate 0.2 \
                    --drop_attn_rate 0.05 \
                    --hungarian_multi \
                    --hung_match_ratio 5 \
                    --coco_path {path_to_coco} \
                    --output_dir output/{scriptname}/{layer_to_det}")

if __name__ == "__main__":
    device = '0,1,2,3,4,5,6,7'
    run_train(device=device, layer_to_det=24)
