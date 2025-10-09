<!--
 * @Author: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 * @Date: 2025-10-06 21:00:18
 * @LastEditors: Zhiwei Zhu (zhuzhiwei21@zju.edu.cn)
 * @LastEditTime: 2025-10-06 21:34:13
 * @FilePath: /UniGSC/third_party/pcc_codec/bin/gpcc/README.md
 * @Description: 
 * 
 * Copyright (c) 2025 by Zhiwei Zhu, All Rights Reserved. 
-->
# The `tmc3` build command 

```bash
git clone https://git.mpeg.expert/MPEG/Explorations/GSC/gsc-software/g-pcc-for-gsc
cd g-pcc-for-gsc
git checkout master-v12.x
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

# The `tmc3_m73982` build command 

```bash
git clone https://git.mpeg.expert/MPEG/Explorations/GSC/gsc-software/g-pcc-for-gsc
cd g-pcc-for-gsc
git checkout master-v12.x
# The patch file can be found in the attachment of proposal m73982
git apply gen-script-gpcc/JEE6.6-Test6-TMC13.patch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```