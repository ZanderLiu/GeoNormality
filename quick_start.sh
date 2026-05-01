#!/bin/bash
# 
# useage:
#   chmod +x quick_start.sh
#   ./quick_start.sh

set -e  
echo "📃 Official Code and Model for our work——"
echo "GeoNormality: 3D Geometric Consistency Distillation for Image Forgery Detection"


echo: "✅ Step1: Download all packages and set configs."
pip install -r requirements.txt # download all packages

echo "✅ Step2: Start Training and validating."

python train_with_depth.py --config config/default.yaml





