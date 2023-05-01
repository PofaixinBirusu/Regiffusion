# Point Cloud Registration with Zero Overlap Rate and Negative Overlap Rate
## Prerequisites
See`requirements.txt`for required packages. Our source code was developed using Python 3.7 with PyTorch 1.7.1+cu101. However, we do not observe problems running on newer versions available as of time of writing.
## Pretrained models
Our pretrained models can be downloaded from [here](https://pan.baidu.com/s/1UTK-rvTggXu2nKKoIEsmMg).
and the extraction code is "tuqb".
## Run the demo to see the effect
```python
python demo.py
```
## Reference Code
Our code also integrates other people's code
`overlap_predator`sourced from [PREDATOR](https://github.com/ShengyuH/OverlapPredator).
`rpmnet`sourced from [RPM-Net](https://github.com/yewzijian/RPMNet).
In the future, we will add the code for [REGTR](https://github.com/yewzijian/RegTR),[GeoTransformer](https://github.com/qinzheng93/GeoTransformer) and [DCP](https://github.com/WangYueFt/dcp).
## Dataset
At present, there are mini versions of Pokemon-Zero and Pokemon-Neg datasets in the code. In the future, we will disclose the complete dataset, including the original stl model.