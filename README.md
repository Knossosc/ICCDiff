# Image Intrinsic Components Guided Conditional Diffusion Model For Low-Light Image Enhancement (ICCDiff) TCSVT2024
This is an official pytorch implement of ICCDiff:

"Image Intrinsic Components Guided Conditional Diffusion Model For Low-Light Image Enhancement" [Paper](https://ieeexplore.ieee.org/document/10633292)

<img src="figs/model.pdf" width="800px">

## Requirements
We implement all the experiments in the following environment.
1. python=3.7.0
2. torch=1.11.0
3. torchvision=0.12.0
4. PIL

## Running

### Testing
Put the test images folder into "./test/IMAGE_FOLDER" or change the option "--test_folder" to your image dir

```
python main.py --state 'eval' --device YOUR_DEVICE --test_folder YOUR_IMAGE_DIR
```

## Citation

If you find ICCDiff is useful in your research, please cite our paper:

'''
@ARTICLE{10633292,
  author={Kang, Sicong and Gao, Shuaibo and Wu, Wenhui and Wang, Xu and Wang, Shuoyao and Qiu, Guoping},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Image Intrinsic Components Guided Conditional Diffusion Model for Low-light Image Enhancement}, 
  year={2024},
  doi={10.1109/TCSVT.2024.3441713}}
'''
