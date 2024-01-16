# FouriDown: Factoring Down-Sampling into Shuffling and Superposing (NeurIPS 2023)

<div align="center"><img src='images/Intro.png' width="80%" height="auto">
></div>

## Framework

FouriDown, as a generic operator, comprises four key components: 2D discrete Fourier transform, context shuffling rules, Fourier weighting-adaptively superposing rules, and 2D inverse Fourier transform. These components can be easily integrated into existing image restoration networks.

<img src='images/Framework.jpg'></img>

## Feature Visualization

The model equipped with FouriDown generates much unique and strong global responses. In contrast, the model with other down-sampling method responds weakly to these regions.

<img src='images/v1.png'></img>

## Feature Spectrum Visualization

Compared to other methods, our FouriDown adaptively adjusts the high and low frequencies, resulting in a wider-band response in the output feature spectrum. Contrasted with previous methods that used fixed frequency aliasing patterns, our approach activates a broader bandwidth on the spectrum, bringing the enhanced performance in image restoration.

<img src='images/v2.png'></img>

## 🎫 Contact

If you have any problems with the released code, please do not hesitate to contact me by email (zqcrafts@mail.ustc.edu.cn).

## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@inproceedings{zhu2023fouridown,
  title={FouriDown: Factoring Down-Sampling into Shuffling and Superposing},
  author={Zhu, Qi and Zhou, Man and Huang, Jie and Zheng, Naishan and Gao, Hongzhi and Li, Chongyi and Xu, Yuan and Zhao, Feng},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


