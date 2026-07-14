<div align="center">
<h1>Uncertainty-Guided Edge Learning for Deep Image Regression in Remote Sensing</h1>

<a href="https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/html/Nguyen_Uncertainty-Guided_Edge_Learning_for_Deep_Image_Regression_in_Remote_Sensing_CVPRW_2026_paper.html"><img src="https://img.shields.io/badge/arXiv-2605.05590-b31b1b" alt="CVPRW 2026"></a>


**[Australian Institute for Machine Learning (AIML)](https://adelaide.edu.au/research/australian-institute-for-machine-learning/)**; **[Adelaide University](https://adelaide.edu.au/)**

[Anh Vu Nguyen](https://anh-vunguyen.github.io/), [Dino Sejdinovic](https://sejdino.github.io/), [Tat-Jun Chin](https://scholar.google.com/citations?user=WyqGF10AAAAJ&hl=en)
</div>

```bibtex
@inproceedings{nguyen2026uncertainty,
  title={Uncertainty-Guided Edge Learning for Deep Image Regression in Remote Sensing},
  author={Nguyen, Anh Vu and Sejdinovic, Dino and Chin, Tat-Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10299--10309},
  year={2026}
}
```

## Updates
- [July 14 2026] requirements.txt is added. README.md is updated.

- [July 1 2026] All refactored files have been added. We are working on cleaning up the environment and preparing a new README file with instructions on how to set it up and use it.

- [Jun 4 2026] A poster has been added. We are still cleaning and optimising the code to enhance readability. All code will be published soon. (We sincerely apologise for the delay, as most of our resources and time are currently focused on the 2 following projects.)


## Quick Start
Please clone this repository to your local machine, and install the dependencies.
```bash
git clone https://github.com/anh-vunguyen/UGEL
cd UGEL
pip install -r requirements.txt
```
An example for running UGEL:
```bash
python run_reg_indtest.py --model beta_mobilenetv3_semi_supervised --lr 0.001 --nQuery 6 --nStart 12 --nEnd 600 --train_path "data/CloudSEN12_128" --test_path "data/CloudSEN12_test128"" --data CloudSEN12_128 --alg uncertain_al_confident_ssl 
``` 

## Acknowledgements
We sincerely thank the authors of these excellent repositories: [BADGE](https://github.com/JordanAsh/badge), [DeepAL](https://github.com/ej0cl6/deep-active-learning)