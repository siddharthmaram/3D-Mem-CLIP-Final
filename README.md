# 3D Mem CLIP Final

## Setup
1. Clone the repository
```bash
git clone https://github.com/siddharthmaram/3D-Mem-CLIP-Final.git
cd 3D-Mem-CLIP-Final
```
2. Create conda environment
```bash
conda create -n 3dmem python=3.9
conda activate 3dmem
```
3. Install necessary packages
```bash
pip install -r requirements.txt
```
4. Install HM3D dataset (val split) and add the path to `cfg/eval_goatbench.yaml`.

## Run
To the the code, use the following command
```bash
python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml
```

For visualizations and videos, change the following in `cfg/eval_goatbench.yaml`.
```yaml
save_visualization: true
save_video: true
```
