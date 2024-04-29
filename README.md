# High-Order Structure Based Middle-Feature Learning for Visible-Infrared Person Re-Identification (AAAI 2024) 




### 1. Prepare the datasets.


- (1) SYSU-MM01 Dataset [1]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

- run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.


- (2) RegDB Dataset [2]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

- A private download link can be requested via sending an email to mangye16@gmail.com. 

- (3) LLCM Dataset [3]: The LLCM dataset can be downloaded from this [website](https://github.com/ZYK100/LLCM) by submitting a copyright form.
- Please send a signed dataset release agreement copy to zhangyk@stu.xmu.edu.cn

### 2. Training and Testing.
Before train the hos-net, you can download the baseline ckpt from [CAJ](https://drive.google.com/drive/folders/107vztbRqim8-oQAuEBh_9S8oTKdhuV2j?usp=sharing) and put it in `./baseline/`  The results might be better by finetuning the hyper-parameters.


```bash
python train_hos_net.py
```

ckpt and log can be seen in `./sle_ckpt/`, `./sle_hsl_ckpt/`, and `./sle_hsl_cfl_ckpt/`  


### 3. Citation

Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{qiu2024high,
  title={High-Order Structure Based Middle-Feature Learning for Visible-Infrared Person Re-Identification},
  author={Qiu, Liuxiang and Chen, Si and Yan, Yan and Xue, Jing-Hao and Wang, Da-Han and Zhu, Shunzhi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4596--4604},
  year={2024}
}
```

###  4. References.

[1] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[2] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[3] Y. Zhang, H. Wang. Diverse embedding expansion network and low-light cross-modality benchmark for visible-infrared person re-identification. In IEEE Computer Vision and Pattern Pecognition (CVPR), pages 2153-2162, 2023.
