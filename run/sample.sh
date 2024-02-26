#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

model_params="/Users/jongbeomkim/Documents/ddpm/kr-ml-test/ddpm_celeba_64×64.pth"
save_dir="/Users/jongbeomkim/Desktop/workspace/ImprovedDDPM/samples/"
img_size=64

python3 ../sample.py\
    --mode="normal"\
    --model_params="$model_params"\
    --save_path="$save_dir/normal/0.jpg"\
    --img_size=$img_size\
    --batch_size=2\
