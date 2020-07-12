# video-object-removal
Just draw a bounding box and you can remove the object you want to remove.
## Installation
All code has been tested on MAC OSX 15.4 Catalina, python 3.5, Pytorch 0.4.0

- Clone the repository 
```shell
git clone https://github.com/zllrunning/video-object-removal.git
cd video-object-removal
cd get_mask
export $PYTHONPATH = $PWD:$PYTHONPATH
bash make.sh
cd ../inpainting
export $PYTHONPATH = $PWD:$PYTHONPATH
bash install.sh
cd ..
```

## Demo
- Download pretrained models of [SiamMask](http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth) and [Inpainting](https://drive.google.com/file/d/1KAi9oQVBaJU9ytr7dYr2WwEcO5NLiJvo/view?usp=sharing)
- Put them in `cp/` folder
- Then just run:
```
python demo.py --data data/Human6
```
- It also **supports video file**.
```
python demo.py --data data/bag.avi
```
- Another optional parameter : `--mask-dilation`

```
python demo.py --data data/Human6  --mask-dilation 24
```
This parameter controls the size of the dilation kernel used for the mask. The role is to expand the range of the mask to avoid edge problems. Please see `inpainting/davis.py` for more details.

---
## Result

![](results/Result.gif)

---

## Acknowledgement
- This repo is based on [SiamMask](https://github.com/foolwood/SiamMask) and [Deep-Video-Inpainting](https://github.com/mcahny/Deep-Video-Inpainting). Many thanks to the excellent repo.

## Citation
```
@article{Wang2019SiamMask,
    title={Fast Online Object Tracking and Segmentation: A Unifying Approach},
    author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
    journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}
```
```bibtex
@inproceedings{kim2019deep,
  title={Deep Video Inpainting},
  author={Kim, Dahun and Woo, Sanghyun and Lee, Joon-Young and So Kweon, In},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5792--5801},
  year={2019}
```












