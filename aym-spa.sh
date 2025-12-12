#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetuneV2.py --config /mnt/storage/ahousenbold/independent-study-aidan/configs/aym-spa.json
