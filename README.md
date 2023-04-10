# OSTrack_SAM
Combining OSTrack and Segment Anything for VOT and VOS

# Some Result
![image](https://github.com/miaodeshui/OSTrack_SAM/blob/main/assets/1.jpg)
![image]([https://github.com/miaodeshui/OSTrack_SAM/blob/main/assets/9.jpg](https://github.com/miaodeshui/OSTrack_SAM/blob/main/assets/cat.gif))


# Quick start

## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n ostrack python=3.8
conda activate ostrack
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f ostrack_cuda113_env.yaml
```

## Install Segment Anything:

```bash
python -m pip install -e segment_anything
```
## Follow the OSTrack to set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```
## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
   And you can set your own datasets
   
## model preperation
- Download the OSTrack model weights from [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing) 

   Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/ostrack`

- Download the checkpoint for segment-anything and grounding-dino:
```bash
cd OSTrack_SAM

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
## Run demo
- Run Demo
```bash
export CUDA_VISIBLE_DEVICES=0
python tracking_sam_demo.py \
   ostrack \
  vitb_384_mae_ce_32x4_got10k_ep100 \
  --dataset got10k_val \
  --threads 4 \
  --num_gpus 1 \
  --checkpoint sam_vit_h_4b8939.pth \
  --sequence (any sequence in datasets)\
```
 ### Thanks other awesome SAM projects:
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Zero-Shot Anomaly Detection](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection)
- [EditAnything: ControlNet + StableDiffusion based on the SAM segmentation mask](https://github.com/sail-sg/EditAnything)
- [IEA: Image Editing Anything](https://github.com/feizc/IEA)
