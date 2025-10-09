<!--
 * @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 * @Date: 2025-07-15 15:27:07
 * @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 * @LastEditTime: 2025-07-15 16:41:07
 * @FilePath: /VGSC/docs/install_codec.md
 * @Description: 
 * 
 * Copyright (c) 2025 by Zhiwei Zhu (zhuzhiwei21@zju.edu.cn), All Rights Reserved. 
-->
## 🔧 Install Core Codecs


### 🔵 Video codec


#### ✅ FFmpeg 
To get started quickly, you can install FFmpeg via your system’s package manager:

```bash
sudo apt-get install ffmpeg
cp $(which ffmpeg) third_party/video_codec/bin/ffmpeg
```
However, for consistent and reproducible results across platforms, we recommend building FFmpeg from source using the same version (7.1.1) as used in our experiments.

<details>
<summary> From Source (Optional)</summary>

```bash
cd third_party/video_codec/src
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg 
git checkout n7.1.1
./configure --enable-gpl --enable-nonfree --enable-libx264 --enable-libx265
make -j$(nproc)
cp ./ffmpeg ../../bin/ffmpeg
```
</details>

#### ✅ HM (Reference Software of HEVC) 
HM is the official reference software for HEVC (High Efficiency Video Coding).
It generally offers better compression performance than FFmpeg but is significantly slower. If you prioritize compression quality, installing HM is recommended.

<details>
<summary>Install HM from Source</summary>

```bash
cd third_party/video_codec/src
git clone https://vcgit.hhi.fraunhofer.de/jvet/HM.git
cd HM 
git checkout HM-18.0
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp ./bin/TAppEncoderStatic ../../bin/hm_encoder
cp ./bin/TAppDecoderStatic ../../bin/hm_decoder
```
</details>

### 🟡 PCC Codecs 
#### ✅ TMC13 (MPEG GPCC)
TMC13 is the reference software for the MPEG Geometry-based Point Cloud Compression standard (GPCC). GPCC can be extended to support Gaussian splats, which treat Gaussian splats as point clouds with additional attributes. 
<details>
<summary>Install TMC13</summary>


```bash
cd third_party/pcc_codecs/src
git clone https://git.mpeg.expert/MPEG/3dgh/g-pcc/software/ce/mpeg-pcc-tmc13.git gpcc
cd gpcc && git checkout mpeg151
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp ./tmc3 ../../bin/gpcc/
```
</details>