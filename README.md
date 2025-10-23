<!--
 * @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 * @LastEditTime: 2025-10-23 12:16:31
 * @Description: README for UniGSC - Unified Gaussian Splats Codec
-->

# 🚀 UniGSC: Unified Gaussian Splats Codec



![Teaser](./assets/teaser.jpg)

**UniGSC** is a highly modular and extensible framework for compressing **static and dynamic Gaussian Splats**, supporting both video and point cloud codecs. It’s designed for **researchers and developers** to quickly prototype, evaluate, and extend Gaussian Splats compression pipelines.


![Pipeline](./assets/pipeline.jpg)
---

## ✨ Highlights

- 🎯 **High-Efficiency Compression** tailored for Gaussian Splats  
- 🧩 **Modular Architecture** enabling seamless algorithm and codec integration  
- 💻 **User-Friendly CLI** and Python API for flexible experimentation  
- 🛠️ **Comprehensive Tools** for logging, visualization, and benchmarking  
- 📦 Support for multiple **core codecs** (e.g., video: FFmpeg, HM; pcc: G-PCC.)  
- 🌐 Built-in support for **MPEG GSC (Gaussian Splats Coding)** workflows 

## 📊 Benchmark
UniGSC provides a **one-stop benchmarking pipeline** for multiple codecs and configurations, enabling easy comparison across experiments. Below are the benchmark results for video-based GSC methods. Our **UniGSC-video_enhanced** achieve **state-of-the-art performance**, , which can be easily reproduced using the scripts provided in the [Quick Start](#-quick-start) section.

<p float="left">
  <img src="assets/rd_curve/mpeg/152/1f_video/bartender/RGB_PSNR.png" width="32%" />
  <img src="assets/rd_curve/mpeg/152/1f_video/breakfast/RGB_PSNR.png" width="32%" />
  <img src="assets/rd_curve/mpeg/152/1f_video/cinema/RGB_PSNR.png" width="32%" />
</p>

For detailed benchmarks and comparisons, please refer to our [Benchmark Report](./docs/benchmark/mpeg152.md).

## ⚙️ Installation
Clone the repo 
```bash
git clone https://github.com/zhuzhiwei99/UniGSC.git  
cd UniGSC
```
Set up environment
```bash
conda create --name UniGSC python=3.10
conda activate UniGSC
```

Please install PyTorch first, you can choose the right command from [Pytorch](https://pytorch.org/get-started/locally/). For example, for CUDA 11.8:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 
```

Install the required packages
```bash
# Install gsc 
pip install .   # Or use `pip install -e .` for editable mode
# Install additional dependencies
cd examples
pip install -r requirements.txt
```


## ⚡ Quick Start

### 📚 Prepare Dataset
Before running the scripts, make sure your dataset is placed in the `data/` directory and follows the recommended structure shown below:

<details>
<summary>Static dataset structure</summary>

```bash
Static_dataset_name/
├── xxx.ply
│
└── colmap_data
    ├── images
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    │   
    ├── sparse(/0)
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    │
    └── (mask)
        ├── 0000.png
        ├── 0001.png
        └── ...
```
</details>


<details>
<summary>Dynamic dataset structure</summary>

```bash
Dynamic_dataset_name/
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
    │   ├── sparse(/0)
    │   │   ├── cameras.bin
    │   │   ├── images.bin
    │   │   └── points3D.bin
    │   │
    │   └── (mask)
    │       ├── 0000.png
    │       ├── 0001.png
    │       └── ...
    │
    └── frame_NNNN
        ├── images
        │   ├── 0000.png
        │   ├── 0001.png
        │   └── ...
        ├── sparse(/0)
        │   ├── cameras.bin
        │   ├── images.bin
        │   └── points3D.bin
        │
        └── (mask)
            ├── 0000.png
            ├── 0001.png
            └── ...
```
</details>

### ▶️ Run Video-based Gaussian Splats Codec
We have provided example scripts to quickly benchmark the video-based Gaussian Splats Codec using different codecs, e.g., **FFmpeg**, **HM**. It is recommended to use the provided configuration files in `configs/` to specify codec settings. For detailed configuration options, please refer to the [Configuration Files](#️-configuration-files) section.


Run the UniGSC framework with **FFmpeg** as the video codec. 
#### One-Stop Benchmark
```bash
bash scripts/benchmark.sh 1 gsc_dynamic data/GSC_splats/m71763_bartender_stable/colmap_data data/GSC_splats/m71763_bartender_stable/track vgsc configs/ffmpeg/anchor_0.0
```

<details>
<summary>📁 Results Overview</summary>

Each rate point result includes the following files and directories:

| Name | Type | Description |
|------|------|-------------|
| `compression` | Directory | Compressed bitstreams produced by the codec. |
| `config` | Directory | Configuration files for each experiment. |
| `intermediate` | Directory | Temporary files from pre-processing and quantization steps. |
| `log` | Directory | Logs for the full pipeline. |
| `reconstructed` | Directory | Reconstructed Gaussian Splats after decoding. |
| `renders` | Directory | Rendered images from uncompressed/reconstructed data. |
| `stats` | Directory | Evaluation statistics including bitrate, quality, timing, and plots. |
| `yuv` | Directory | Intermediate YUV video files used as codec input/output. |
| `rd_summary_GT.json` | File | Rate-Distortion metrics using ground truth images as reference. |
| `rd_summary_rendered.json` | File | Rate-Distortion metrics using uncompressed Gaussian Splats renders as reference. |
</details>


#### Run Steps Individually
| Step | Command |
|------|---------|
| 🗜️ **Encode** | ```bashbash scripts/encode.sh 1 data/GSC_splats/m71763_bartender_stable/track vgsc configs/ffmpeg/anchor_0.0``` |
| 🔄 **Decode** | ```bashbash scripts/decode.sh 1 data/GSC_splats/m71763_bartender_stable/track vgsc configs/ffmpeg/anchor_0.0``` |
| 📊 **Evaluate** | ```bashbash scripts/eval.sh 1 gsc_dynamic data/GSC_splats/m71763_bartender_stable/colmap_data data/GSC_splats/m71763_bartender_stable/track results/GSC_splats/m71763_bartender_stable/track/frame1/configs/ffmpeg/anchor_0.0``` |

💡 **Tips:**
1. If you want to get better RD performance, you can use `HM` as the video codec by replacing `ffmpeg` with `hm` in the above commands.
2. For each script, you can add `--help` to see all available options.


### ▶️ Run PCC-based Gaussian Splats Codec
UniGSC also supports point cloud codecs, e.g., MPEG GPCC (TMC13), allowing Gaussian Splats to be compressed as point clouds.

Benchmark the MPEG GPCC-based GSC codec using:

```bash
bash scripts/benchmark.sh 1 gsc_dynamic data/GSC_splats/m71763_bartender_stable/colmap_data data/GSC_splats/m71763_bartender_stable/track gpcc configs/gpcc/mpeg/152/jee6.6
```




### 🌐 One-Stop MPEG GSC Benchmark Platform
If you are interested in MPEG GSC workflows, we have prepared dedicated scripts to facilitate benchmarking different MPEG GSC methods on both the **I-3DGS Main Track** and the **I-3DGS 1-Frame Track**.
 
The full pipeline covers **🗜️ Encoding → 🔄 Decoding → 🎨 Rendering → 📊 Evaluation**.  

💡 **Tips:** Render uncompressed views first — they serve as references, so you won’t need to re-render them for each configuration.


#### 🎬 I-3DGS Main Track

| Task | Command |
|------|---------|
| Render uncompressed Gaussian Splats | `bash scripts/mpeg/main_render.sh` |
| Benchmark **gpcc_ctc** | `bash scripts/mpeg/main_benchmark.sh gpcc configs/gpcc/mpeg/152/jee6.2` |
| Benchmark **gpcc_enhanced** | `bash scripts/mpeg/main_benchmark.sh gpcc configs/mpeg/151/gpcc/m73385_octree-predlift/Combined_Predlift` |
| Benchmark **video_ctc** | `bash scripts/mpeg/main_benchmark.sh vgsc configs/mpeg/152/video/video_ctc/` |
| Benchmark **video_enhanced** | `bash scripts/mpeg/main_benchmark.sh vgsc configs/mpeg/152/video/video_enhanced` |




#### 🎬 I-3DGS 1-Frame Track

| Task | Command |
|------|---------|
| Render uncompressed Gaussian Splats | `bash scripts/mpeg/1f_forward_facing_render.sh` |
| Benchmark **gpcc_ctc** | `bash scripts/mpeg/1f_forward_facing_benchmark.sh gpcc configs/gpcc/mpeg/152/jee6.6` |
| Benchmark **gpcc_enhanced** | `bash scripts/mpeg/1f_forward_facing_benchmark.sh gpcc configs/mpeg/151/gpcc/m73385_octree-predlift/Combined_Predlift` |
| Benchmark **video_ctc** | `bash scripts/mpeg/1f_forward_facing_benchmark.sh vgsc configs/mpeg/152/video/video_ctc/` |
| Benchmark **video_enhanced** | `bash scripts/mpeg/1f_forward_facing_benchmark.sh vgsc configs/mpeg/152/video/video_enhanced` |

💡 For **object_centric sequences**, replace `forward_facing` with `object_centric` in the script filenames.

---



## 🧪 Usage Guide
### 📁 Project Structure

| Directory      | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `assets/`      | Demo assets, such as teaser images                                          |
| `docs/`        | Documentation and installation guides                                       |
| `gsc/`        | Core framework modules (codecs, mapping, preprocessing, quantization, etc.)  |
| `examples/configs/`     | Configuration files for different compression settings                      |
| `examples/data/`        | Gaussian Splats datasets (PLY point clouds and COLMAP data)                  |
| `examples/datasets/`    | Dataset handling and loading scripts                                        |
| `examples/results/`     | Encoded bitstreams, rendered views, evaluation metrics, etc.                |
| `examples/scripts/`     | Helper scripts for automating workflows                                     |
| `examples/third_party/` | External dependencies (e.g., FFmpeg, HM, GPCC, fine-tuning tools)           |
| `examples/utils/`       | Utility functions (I/O, metrics, plotting, summaries, etc.)                 |
| `examples/gs_pipeline.py` | Main entry point for running experiments with different codecs and settings |

### 🧱 Install Core codecs
As a Unified Gaussian Splats Coding framework, UniGSC requires core codecs for compression. We currently support FFmpeg, HM, G-PCC. 

We have provided prebuilt binaries for FFmpeg, HM and GPCC under `third_party/video_codec/bin/` directory, which you can use directly. If you prefer to build them from source, please refer to the [Codecs Installation Guide](./docs/install_codec.md).

To use custom codecs, place your encoder and decoder binaries in `third_party/video_codec/bin/` and and update the configuration accordingly.

### ⚙️ Configuration Files
UniGSC supports multiple codecs. You must specify the codec type and its corresponding settings via either command-line arguments or a `YAML` configuration file. The description of each command-line arguments can be found by running:
```bash
python gs_pipeline.py vgsc --help
python gs_pipeline.py gpcc --help
```

We recommend using `YAML` for better organization and maintainability. For complex experiments, you can create separate configuration files for different settings. 

Configuration files are located in the `configs/` directory. Example configurations:
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
  trans_quats_norm: true
  trans_sh0_ycbcr: false
  trans_shN_pca: true
  trans_shN_rank: 21

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
  encoder_path: third_party/pcc_codec/bin/gpcc/tmc3
  decoder_path: third_party/pcc_codec/bin/gpcc/tmc3
  encode_config_path: third_party/pcc_codec/cfg/gpcc/mpeg/152/jee6.6/r01/encoder.cfg
  decode_config_path: third_party/pcc_codec/cfg/gpcc/mpeg/152/jee6.6/r01/decoder.cfg

  # Quantize options
  quant_type: pcc_N01292
```
</details>


You can start with the provided configuration files and modify them according to your needs. 

- FFmpeg: `configs/ffmpeg/anchor_0.0/rp04.yaml` 
- HM:  `configs/hm/anchor_0.0/rp04.yaml` 

If you are interested in MPEG GSC workflows, please refer to the configurations under `configs/mpeg`. For example:
- video_ctc: `configs/mpeg/152/video/video_ctc/rp04.yaml`
- gpcc_ctc: `configs/mpeg/152/gpcc/jee6.6/r04.yaml`




### 💻 Command-Line Interface
We provide command-line scripts to encode, decode, render, and evaluate Gaussian Splats with different codecs:
| Script              | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `gsc/runner.py`      | Defines the core workflow: pre_process, quantize, encode, decode, dequantize, post_process, render, evaluate of Gaussian Splats. |
| `examples/gs_pipeline.py`    | Entry point for running experiments with different pipeline types, e.g.: benchmark, codec, encode, decode, pre-process, quantize, render, eval.                                 |



🎯 Supported GSCodec Types
- `vgsc`: Video-based Gaussian Splats Codec 
- `gpcc`: GPCC-based Gaussian Splats Codec

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


## 🐍 Using the UniGSC Python API
In short, the UniGSC Python API allows researchers and developers to:

* Load Gaussian Splats datasets
* Apply pre-processing, including pruning, transformation, etc.
* Perform flexible quantization with customizable bit depth and channel control
* Encode and decode Splats with video or point cloud codecs
* Render and evaluate reconstructed results

<details> <summary> <code>UniGSC API Usage Examples</code> </summary>

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

# Load Gaussian Splats
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



## 📄 Citation

If you find this project useful, please consider giving it a ⭐.

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
