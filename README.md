# MM-Net-PyTorch (IEEE, 2022)
>> code for MM-Net: Multiframe and Multimask-Based Unsupervised Deep Denoising for Low-Dose Computed Tomography, IEEE.
>> We used the clinical dataset of the 2016 NIH-AAPM-Mayo Clinic Low-Dose CT Grand Challenge
>> https://ieeexplore.ieee.org/document/9963593
>>


I have created a GitHub repository to share the code for the paper 'MM-Net: Multiframe and Multimask-Based Unsupervised Deep Denoising for Low-Dose Computed Tomography.' I am currently working on it. 
If you are interested in sharing the code, please feel free to contact me at sunyounge_@ewhain.net. I will update it soon.
## Overall architecture
### Two-step training network 
<img src="https://github.com/sunyoungIT/MM-Net/assets/51948046/73c2d380-6998-409b-bf4e-28bf84ac46da" width="600" height="300"/>

## First Training Step :
### Multiscale Attention U-Net 
The code for attention U-Net can be found at https://github.com/LeeJunHyun/Image_Segmentation. You can find more detailed networks available there. 
<img src="https://github.com/sunyoungIT/MM-Net/assets/51948046/f2632b7c-1b0d-4841-b306-6a7acab1b784" width="700" height="400"/>

## Second Training Step :
### Multipatch and Multi-mask 
<img src="https://github.com/sunyoungIT/MM-Net/assets/51948046/e43a6036-6dbd-4473-aef4-de1aaa3f40f9" width="900" height="300"/>

# Citation 
You may cite this project as:
```
@ARTICLE{9963593,
  author={Jeon, Sun-Young and Kim, Wonjin and Choi, Jang-Hwan},
  journal={IEEE Transactions on Radiation and Plasma Medical Sciences}, 
  title={MM-Net: Multiframe and Multimask-Based Unsupervised Deep Denoising for Low-Dose Computed Tomography}, 
  year={2023},
  volume={7},
  number={3},
  pages={296-306},
  doi={10.1109/TRPMS.2022.3224553}}
```
