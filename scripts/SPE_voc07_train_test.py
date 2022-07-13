import os
import random

def run_train(device='0', layer_to_det=24):
    path_to_voc = 'data/' # path to VOC07, note the data should in coco format
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch --master_port {port}\
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main.py \
                    --dataset_file voc07 \
                    --layer_to_det {layer_to_det} \
                    --backbone TSCAM_cait_XXS36_Two_Branch \
                    --hungarian_multi \
                    --hung_match_ratio 5 \
                    --box_jitter 0.1 \
                    --coco_path  {path_to_voc} \
                    --output_dir output/voc07_exp")


def run_test(device='1'):
    assert len(device) == 1, 'only single gpu testing is implemented, testing with multi-gpu is coming soon'
    path_voc_07_test = 'data/' # path to VOC07 devkit set, not that the data should be in VOC format
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch --master_port {port}\
                --nproc_per_node={num_devices} \
                --use_env \
                test_det_voc.py \
                --backbone TSCAM_cait_XXS36_Two_Branch \
                --resume  output/voc07_exp/checkpoint0049.pth\
                --eval \
                --output_dir output/voc07_exp \
                --coco_path {path_voc_07_test} ")

                
if __name__ == "__main__":
    device = '0,1,2,3,4,5,6,7'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = 'GNU'
    run_train(device=device)
    run_test()