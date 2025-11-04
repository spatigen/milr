# MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning

<div align="center">

[**Yapeng Mi**](https://scholar.google.com/citations?user=xr7kNGEAAAAJ&hl=zh-CN),
[**Hengli Li**](https://scholar.google.com/citations?user=K7gsqkMAAAAJ&hl=en),
[**Yanpeng Zhao**](https://scholar.google.com/citations?user=-T9FigIAAAAJ&hl=en),
[**Chenxi Li**](https://openreview.net/profile?id=~Chenxi_Li7),
[**Huimin Wu**](https://scholar.google.com/citations?user=9HH9I6YAAAAJ&hl=en),
[**Xiaojian Ma**](https://jeasinema.github.io/),
[**Song-Chun Zhu**](https://scholar.google.com/citations?user=Al8dyb4AAAAJ&hl=en),
[**Ying Nian Wu**](https://scholar.google.com/citations?user=7k_1QFIAAAAJ&hl=en),
[**Qing Li**](https://liqing.io/)

[\[🌐 Project Page\]](https://spatigen.github.io/milr.io/) [\[📜 Paper\]](https://www.arxiv.org/abs/2509.22761)
</div>

![teaser map](fig/teaser.png)

We introduce MILR, a test-time method that performs joint reasoning over text and image in a unified latent space. It searches over vector representations of discrete text/image tokens using policy gradients guided by an image-quality critic, instantiated within the MUG framework (which supports language reasoning before image synthesis). MILR optimizes intermediate model outputs as the latent space, requiring no fine-tuning and operating entirely at inference time.

## Installation

```bash
conda create -n latentseek python=3.10
conda activate latentseek
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

#install Geneval configs
#You may meet package counters, it doesn't matter
pip install -U openmim
mim install mmengine mmcv-full==1.7.2
cd src/geneval
./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .

cd ../rewards
./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"
```
If you meet error with flash_attn==2.7.2.post1, you can refer to the `https://github.com/Dao-AILab/flash-attention/releases` to download.

## Usage
We support different kinds of reward types. And we test on three benchmarks: **Geneval**, **T2I-CompBench**, **Wise**

### Geneval

```bash
cd src
bash scripts/geneval_both.sh
```

The bash file

```bash
#!/bin/bash
PATH_TO_DATA="prompts/geneval/evaluation_metadata.jsonl"
PATH_TO_MODEL="deepseek-ai/Janus-Pro-7B"
output_dir="./geneval_results/long_results" #self create the dir
optimize_mode="both"  # or "image"
reward_model_type="geneval"
text_k=0.1 
image_k=0.01 
lr=0.01
max_text_steps=30
max_image_steps=30
max_both_steps=30

# === set log file name ===
if [ "$optimize_mode" = "text" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_lr${lr}_ts${max_text_steps}.txt"
elif [ "$optimize_mode" = "image" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_ik${image_k}_lr${lr}_is${max_image_steps}.txt"
else
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_ik${image_k}_lr${lr}_bs${max_both_steps}.txt"
fi

# === train script ===
CUDA_VISIBLE_DEVICES=1 python main_janus.py \
    --dataset "$PATH_TO_DATA" \
    --model_name_or_path "$PATH_TO_MODEL" \
    --output_dir "$output_dir" \
    --optimize_mode "$optimize_mode" \
    --reward_model_type "$reward_model_type" \
    --lr "$lr" \
    --text_k "$text_k" \
    --image_k "$image_k" \
    --max_text_steps "$max_text_steps" \
    --max_image_steps "$max_image_steps" \
    --max_both_steps "$max_both_steps" \
    --device "cuda" \
    > "$LOG_FILE" 2>&1 &
```
- `optimize_mode`: The mode of optimization, you can choose from `both`, `image` or `text`.
- `reward_model_type`: the reward model used for optimize, you can check in the main_janus.py file
- `text_k`: the ratio of text tokens for optimization
- `image_k`: the ratio of image tokens for optimization
- `lr`: the learning rate
- `max_text_steps`: the steps of text optimization
- `max_image_steps`: the steps of image optimization
- `max_both_steps`: the steps of both optimization

### Different Rewards
For `SelfReward`, you can run the script:
```bash
bash scripts/geneval_self_reward.sh 
```
For `UnifiedReward`, you can run the script:
```bash
bash scripts/geneval_unified_reward.sh 
```
remeber you download the `CodeGoat24/UnifiedReward-qwen-7b` model.

For `MixedReward`, you can run the following script:
```bash
cd rewards/MixedReward
git clone https://github.com/IDEA-Research/GroundingDINO.git ## follow the guide of https://github.com/IDEA-Research/GroundingDINO
mkdir reward_weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
huggingface-cli download microsoft/git-large-vqav2 --repo-type model --local-dir git-large-vqav2
cd ../..
bash scripts/unified_reward_geneval.sh 
```

For `GPT-4o` reward, you should first set the api key in `main_janus.py`, and run the following script:
```bash
bash scripts/geneval_gpt4o_reward.sh 
```


### T2I-CompBench
For T2I-CompBench, we use this benchmark from `https://github.com/Karine-Huang/T2I-CompBench`
And you should follow the guidence of the repo to install the environment, and put the weights into the `scripts/T2ICompBench`

then you can run the scripts in `scripts/T2I-CompBench` as follows:
```bash
bash scripts/T2I-CompBench/MetricReward_T2ICompBench_color.sh 
```
### WISE
For WISE, we use this benchmark from `https://github.com/PKU-YuanGroup/WISE`.
You also should install the environment following the guidences and fill the right api key into `rewards/wise_reward.py`.

Then you can run the script as follows:
```bash
bash scripts/Wise/wise_reward_wise_cultural_common_sense.sh 
```

## Citation
```bibtex
@article{mi2025milr,
  title={MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning},
  author={Mi, Yapeng and Li, Hengli and Zhao, Yanpeng and Li, Chenxi and Wu, Huimin and Ma, Xiaojian and Zhu, Song-Chun and Wu, Ying Nian and Li, Qing},
  journal={arXiv preprint arXiv:2509.22761},
  year={2025}
}
```

## Contact
* If you have any questions, please send me an email at: miyapeng78@gmail.com
