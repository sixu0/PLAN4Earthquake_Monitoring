<p align="center" width="100%">
<img src="assets\PLAN.png"  width="80%" height="80%">
</p>


<div>
<div align="center">
    <a href='https://sixu0.github.io/' target='_blank'>Xu Si<sup>1</sup></a>&emsp;
    <a href='http://cig.ustc.edu.cn/people/list.htm' target='_blank'>Xinming  Wu<sup>1,†,‡</sup></a>&emsp;
    <a href='https://dams.ustc.edu.cn/main.htm' 
    target='_blank'>Zefeng Li<sup>1,†</sup></a>&emsp;
    </br>
    Shenghou Wang<sup>2</sup></a>&emsp;
    <a href='https://dams.ustc.edu.cn/main.htm' 
    target='_blank'>Jun Zhu<sup>1</sup></a>&emsp;

</div>
<div>

<div align="center">
    <sup>1</sup>
    University of Science and Technology of China&emsp;
    </br>
    <sup>2</sup>
    China University of Geosciences&emsp;
    </br>
    <!-- <sup>*</sup> Equal Contribution&emsp; -->
    <sup>†</sup> Corresponding Author&emsp;
    <sup>‡</sup> Project Lead&emsp;
</div>

-----------------

[![arXiv](https://img.shields.io/badge/arxiv-2306.13918-b31b1b?style=plastic&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2306.13918)](https://arxiv.org/abs/2306.13918)
[![CEE](https://img.shields.io/badge/CEE-22(2024)-ced870)](https://www.nature.com/articles/s43247-023-01188-4)
![GitHub followers](https://img.shields.io/github/followers/sixu0?style=social)
![GitHub stars](https://img.shields.io/github/stars/sixu0/PLAN4Earthquake_Monitoring?style=social)


### 🌟 An all-in-one seismic Phase picking, Location, and Association Network


 <!-- As shown in this figure, PLAN can achieve phase picking, location and phase association  -->


# 🌟 News
* **2024.4.28:**  The code for training were released. More comprehensive and detailed tutorial, will be gradually released. 
* **2024.1.31:**  The Jupyter tutorial and model weights were released. 
* **2024.1.6:**  🌟🌟🌟 Congratulation! The paper has been published on Communication Earth & Environment [Links](https://doi.org/10.1038/s43247-023-01188-4). The code will be released before January 31, 2024.
* **2023.6.24:** Paper is released at [arxiv](https://arxiv.org/abs/2306.13918), and code will be gradually released.
* **2023.4.10:** Github Repository Initialization. (copy README template from Meta-Transformer)


# How to Use

##### 1. Via Anaconda and pip (recommended):
    conda create -n PLAN python==3.11.0
    pip install -r requirements.txt
    conda activate PLAN
    
##### PS1: If there are some problem, you can use pip tsinghua channels. 
    
##### PS2: If there some problem about net, you can also install lib one by one by pip. Some lib is needed (Torch,PyG,Numpy,Pandas,Geopy,tqdm,tensorboard.)

##### 2. Download Weights ([Google Drive](https://drive.google.com/file/d/1OX1IE6Oh5AsOSXjdlAShVzWK0u3bGV9G/view)) file and put it 'model' dir.
##### 3. Download sac file and put it 'data/1h_data/' dir. You can download it through [STP](https://scedc.caltech.edu/data/waveform.html) or [Google Drive](https://drive.google.com/file/d/1_VmMz0gapc6oEYOIKoBZJWg0ulMgR-2a/view).
##### 4. Open Tutorial_PLAN.ipynb in Notebook dir.

&ensp;
# Citation
If the code and paper help your research, please kindly cite:
```
@article{Si2024PLAN,
  title={An all-in-one seismic phase picking, location, and association network for multi-task multi-station earthquake monitoring},
  author={Si, Xu and Wu, Xinming and Li, Zefeng and Wang, Shenghou and Zhu, Jun},,
  journal={Communications Earth \& Environment},
  volume={5},
  number={1},
  pages={22},
  year={2023},
  publisher={Nature Publishing Group UK London},
  doi = {10.1038/s43247-023-01188-4},
}
```
# License
This project is released under the [MIT license](LICENSE).

# Acknowledgement
This code is developed based on excellent open-sourced projects including [PhaseNet](https://github.com/AI4EPS/PhaseNet), [EQTransformer](https://github.com/smousavi05/EQTransformer),  [Seisbench](https://github.com/seisbench/seisbench) and [STEAD](https://github.com/smousavi05/STEAD).
