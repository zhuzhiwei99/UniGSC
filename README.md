<!--
 * @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 * @LastEditTime: 2025-10-01 10:09:02
 * @Description: README for UniGSC - Video-based Gaussian Splat Coding
-->

# 🚀 UniGSC: Unified Gaussian Splat Coding Platform

<!-- [![CI](https://github.com/zhuzhiwei99/UniGSC/actions/workflows/ci.yml/badge.svg)](https://github.com/zhuzhiwei99/UniGSC/actions)
[![PyPI version](https://img.shields.io/pypi/v/vgsc.svg)](https://pypi.org/project/vgsc/)
[![Python versions](https://img.shields.io/pypi/pyversions/vgsc.svg)](https://pypi.org/project/vgsc/) -->

![Teaser](./assets/teaser.jpg)


**UniGSC** is a highly modular and extensible framework for compressing **static and dynamic Gaussian splats**, supporting both video and point cloud codecs. It’s designed for **researchers and developers** to quickly prototype, evaluate, and extend Gaussian Splat compression pipelines.

---

## ✨ Highlights

- 🎯 **High-Efficiency Compression** for Gaussian splats
- 🧩 **Modular Architecture** for algorithm and codec integration
- 📦 Support for multiple **core codecs** (e.g., FFmpeg, HM, G-PCC)
- 📈 Built-in **evaluation tools** (LPIPS, PSNR, etc.)
- 📜 Support for **MPEG GSC (Gaussian Splat Coding)** workflows

---

## 📊 Benchmark
UniGSC provides a **one-stop benchmarking pipeline** for multiple codecs and configurations, enabling easy comparison across experiments. Below we show RD curves on the MPEG GSC dataset using different codecs and settings. We achieve **state-of-the-art performance** on multiple sequences.
<details>
<summary>Commands to reproduce results on the <em>bartender</em> sequence:</summary>

- Generate `MPEG GPCC JEE6.6 ` results using:
```bash
bash scripts/benchmark_with_configs.sh 1 bartender gpcc configs/gpcc/mpeg151/jee6.6
```
- Generate `MPEG Video-based GSC` results using:
```bash
bash scripts/benchmark_with_configs.sh 1 bartender vgsc configs/mpeg/151/video/video_anchor_ctc/
```
- Generate `UniGSC-VGSC` results using:
```bash
bash scripts/benchmark_with_configs.sh 1 bartender vgsc configs/mpeg/152/video/UniGSC-VGSC
```
Other datasets can be processed in the same way by replacing *bartender* with the target sequence name.
</details>

<p float="left">
  <img src="assets/rd_curve/bartender/RGB_PSNR.png" width="30%" />
  <img src="assets/rd_curve/breakfast/RGB_PSNR.png" width="30%" />
  <img src="assets/rd_curve/cinema/RGB_PSNR.png" width="30%" />
</p>


## ⚙️ Installation
Clone the repo 
```bash
git clone https://github.com/zhuzhiwei99/UniGSC.git  
cd UniGSC
```
Set up environment
```bash
conda create --name gsc python=3.10
conda activate gsc
```

Please install PyTorch first, you can choose the right command from [Pytorch](https://pytorch.org/get-started/locally/). For example, for CUDA 11.8:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 
```

Install the required packages
```bash
# Install UniGSC 
pip install .   # Or use `pip install -e .` for editable mode
# Install additional dependencies
pip install -r requirements.txt
```

---

## ⚡ Quick Start
Before running the scripts, please ensure your dataset is correctly placed under the `data/` directory with the structure shown below:
<details>
<summary>Dataset structure</summary>

```bash
DATASET_NAME/
├── plys
│   ├── frame_0000.ply
│   ├── frame_0001.ply
│   ├── ...
│   └── frame_NNNN.ply
│
└── colmap_data
    ├── frame_0000
    │   ├── images
    │   │   ├── 0000.png
    │   │   ├── 0001.png
    │   │   └── ...
    │   │   
    │   └── sparse
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    │
    └── frame_NNNN
        ├── images
        │   ├── 0000.png
        │   ├── 0001.png
        │   └── ...
        └── sparse
            ├── cameras.bin
            ├── images.bin
            └── points3D.bin
```
</details>



### ▶️ Run UniGSC-VGSC (Video-based Gaussian Splats Codec) 
Below is a simple example of running the UniGSC framework with FFmpeg as the video codec.
You can switch to other codecs such as **HM** by replacing `ffmpeg` with `hm` in the commands.
FFmpeg offers faster performance, while HM provides better compression at the cost of speed.

We recommend using the provided configuration files in the `configs/` directory to specify codec settings.

We have provided functional scripts to run the encoding, decoding, rendering, and evaluation steps. For each script, you can simply run `bash scripts/<script_name>.sh` to get the help message.

#### 🚀 One-Click Compression & Evaluation
To run the full compression and evaluation pipeline (including encoding, decoding, rendering, and evaluation using PSNR, SSIM, LPIPS, and MPEG GSC metrics (RGB_PSNR, YUV_PSNR, YUV_SSIM, etc.)), simply use:
```bash
bash scripts/benchmark_with_configs.sh 1 bartender vgsc configs/ffmpeg/anchor_0.0
```
After running this command, you will find the results in the `results/m71763_bartender_stable/track/frame1/configs/ffmpeg/anchor_0.0` directory, including: intermediate yuv files, compressed bitstream, rendered views, and evaluation metrics.
#### 🔧 Run Each Step Separately
You can also execute each step individually:
- Encode

```bash
bash scripts/encode_with_configs.sh 1 bartender vgsc configs/ffmpeg/anchor_0.0
```

- Decode

```bash
bash scripts/decode_with_configs.sh 1 bartender vgsc configs/ffmpeg/anchor_0.0
```

- Evaluate

```bash
bash scripts/eval_with_result_dirs.sh 1 bartender results/m71763_bartender_stable/track/frame1/configs/ffmpeg/anchor_0.0
```
- Render 

```bash
bash scripts/render_with_1_ply_dir.sh 1 bartender data/GSC_splats/m71763_bartender_stable/track
```



### ▶️ Run UniGSC-GPCC (PCC-based Gaussian Splats Codec)
UniGSC also includes wrappers for point cloud codecs, such as the GSC extension of MPEG GPCC, allowing Gaussian splats to be compressed as point clouds.

To run the MPEG GPCC codec (TMC13), use the following command:

```bash
bash scripts/benchmark_with_configs.sh 1 bartender gpcc configs/gpcc/mpeg151/jee6.6
```

---



## 📁 Project Structure

| Directory      | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `assets/`      | Demo assets, such as teaser images                                          |
| `configs/`     | Configuration files for different compression settings                      |
| `data/`        | Gaussian Splat datasets (PLY point clouds and COLMAP data)                  |
| `datasets/`    | Dataset handling and loading scripts                                        |
| `docs/`        | Documentation and installation guides                                       |
| `results/`     | Encoded bitstreams, rendered views, evaluation metrics, etc.                |
| `scripts/`     | Helper scripts for automating workflows                                     |
| `third_party/` | External dependencies (e.g., FFmpeg, HM, GPCC, fine-tuning tools)           |
| `utils/`       | Utility functions (I/O, metrics, plotting, summaries, etc.)                 |
| `gsc/`        | Core framework modules (codecs, mapping, preprocessing, quantization, etc.) |


---



## 🧪 Usage Guide

### 🧱 Install Core codecs
As a Unified Gaussian Splat Coding framework, UniGSC requires core codecs for compression. We currently support FFmpeg, HM, G-PCC. 

We have provided prebuilt binaries for FFmpeg, HM and GPCC under `third_party/video_codec/bin/` directory, which you can use directly. If you prefer to build them from source, please refer to the [installation guide](./docs/install_codec.md).

To use custom codecs, place your encoder and decoder binaries in `third_party/video_codec/bin/` and and update the configuration accordingly.

### ⚙️ Configuration Files
UniGSC supports multiple codecs. You must specify the codec type and its corresponding settings via either command-line arguments or a YAML configuration file. The description of each command-line arguments can be found by running:
```bash
python gs_pipeline.py vgsc --help
python gs_pipeline.py gpcc --help
```

We recommend using YAML for better organization and maintainability. For complex experiments, you can create separate configuration files for different settings. 

Configuration files are located in the `configs/` directory. Example configurations include:
<details> <summary>📄 <code>vgsc_config_example.yaml</code> </summary>
This example shows a typical configuration for UniGSC using FFmpeg. Not all fields are required—if omitted, default values will be used. You can override any of them via command-line arguments when running the scripts.

```yaml
codec: 
  video_codec_type: ffmpeg
  encoder_path: third_party/video_codec/bin/ffmpeg
  decoder_path: third_party/video_codec/bin/ffmpeg
  encode_config_path: null
  decode_config_path: null

  # Prune options
  prune_type: threshold
  prune_thres_opacities: -4
  
  # Transform options
  use_quats_norm: true
  use_sh0_ycbcr: false
  use_shN_pca: true
  shN_rank: 21

  # Quantize options
  quant_type: video_N01292
  quant_per_channel: true
  quant_shN_per_channel: false

  # Sorting options
  sort_type: plas
  sort_with_shN: true

  # YUV Chroma sub/up-sampling methods
  chroma_sub_method: average_pool
  chroma_up_method: bilinear
  
  # Video codec options
  gop_size: 16
  all_intra: false

  # QP config for each attribute
  qp_config:
    means_l: -1
    means_u: -1
    opacities: 7
    quats_w: 2
    quats_xyz: 2
    scales: 7
    sh0: 7
    shN_sh1: 7
    shN_sh2: 12
    shN_sh3: 17

  # Bit depth config for each attribute
  bit_depth_config:
    means_l: 8
    means_u: 8
    opacities: 10
    quats: 10
    scales: 10
    sh0: 10
    shN: 10

  # Pixel format for each attribute
  pix_fmt_config:
    means_l: yuv444p
    means_u: yuv444p
    opacities: yuv400p
    quats_w: yuv400p
    quats_xyz: yuv444p
    scales: yuv444p
    sh0: yuv444p
    shN: yuv444p

```
</details>


<details> <summary>📄 <code>gpcc_config_example.yaml</code> </summary>
This configuration is used for the GPCC codec (TMC13). All five fields are required.

```yaml
codec:
  quant_type: N00677
  encoder_path: third_party/pcc_codec/bin/gpcc/tmc3
  decoder_path: third_party/pcc_codec/bin/gpcc/tmc3
  encode_config_path: third_party/pcc_codec/cfg/gpcc/mpeg151/jee6.6/r01/encoder.cfg
  decode_config_path: third_party/pcc_codec/cfg/gpcc/mpeg151/jee6.6/r01/decoder.cfg
```
</details>


You can start with the provided configuration files and modify them according to your needs. 

- FFmpeg: `configs/ffmpeg/anchor_0.0/rp04.yaml` 
- HM:  `configs/hm/anchor_0.0/rp04.yaml` 

If you are interested in MPEG GSC workflows, please refer to the configurations under `configs/mpeg`. For example:
- Video-based GSC: `configs/mpeg/151/video/video_anchor_ctc/rp04.yaml`
- GPCC-based GSC: `configs/mpeg/151/gpcc/jee6.6/r04.yaml`


### 🐍 Using the UniGSC Python API
In short, the UniGSC Python API allows researchers and developers to:

* Load and preprocess Gaussian splat datasets
* Apply pruning, transforms, and PCA-based SH compression
* Perform flexible quantization with customizable bit depth and channel control
* Encode and decode splats with video or point cloud codecs
* Render and evaluate reconstructed results

<details> <summary>📄 <code>UniGSC API Usage Examples</code> </summary>

```python
from gsc.runner import Runner, VgscCodecConfig

# Create a codec configuration
config = VgscCodecConfig(
    ply_dir="data/scene/ply",
    data_dir="data/scene/colmap",
    result_dir="results/scene",
    codec_type="vgsc",
    video_codec_type="ffmpeg",
    encoder_path="third_party/video_codec/bin/ffmpeg",
    decoder_path="third_party/video_codec/bin/ffmpeg",
    frame_num=1,
)

# Initialize the UniGSC pipeline
runner = Runner(
  local_rank=0,
  world_rank=0,
  world_size=1,
  cfg=config)

# Load Gaussian splats
runner.load_ply_sequence()
# Preprocess 
runner.preprocess()
# Quantize 
runner.quantize()
# Encode 
runner.encode()
# Decode
runner.decode()
# Dequantize
runner.dequantize()
# Postprocess
runner.postprocess()
# Render and evaluate
runner.eval()

```
---</details>

### 💻 Command-Line Interface
We provide command-line scripts to encode, decode, render, and evaluate Gaussian splats with different codecs:
| Script              | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `gsc/runner.py`      | Defines the core workflow: pre_process, quantize, encode, decode, dequantize, and post_process of Gaussian splats |
| `gs_pipeline.py`    | Entry point for running experiments with different pipelines                                    |



🎯 Supported GSCodec Types
- `vgsc`: Video-based Gaussian Splat Coding 
- `gpcc`: GPCC-based Gaussian Splat Coding 

▶️ Example Command

```bash
CUDA_VISIBLE_DEVICES=0 python gs_pipeline.py \
    vgsc \
    --config configs/ffmpeg/anchor_0.0.yaml \
    --pipe_stage benchmark \
    --ply_dir ./data/scene/ply \
    --data_dir ./data/scene/colmap \
    --result_dir ./results/scene_anchor_0.0 \
    --frame_num 50 \
    --test_view_id 0 \
    --codec.gop_size 16
```

ℹ️ Get Help

```bash
python gs_pipeline.py vgsc --help
python gs_pipeline.py gpcc --help
```


---

## 📄 Citation

If you find this project useful, please consider giving it a star.

---

## 👥 Contributors

- Zhiwei Zhu · zhuzhiwei21@zju.edu.cn  
- Sicheng Li · jasonlisicheng@zju.edu.cn  

If you have any questions about this project, please feel free to contact us.

---

## 🙏 Acknowledgements

This project is built upon:

- [GSCodec_Studio](https://github.com/JasonLSC/GSCodec_Studio)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [gsplat](https://github.com/nerfstudio-project/gsplat)

Thanks to all the great open-source contributors! ❤️
