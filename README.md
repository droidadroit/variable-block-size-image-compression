# Variable block-size image compression based on edge detection
I exploited the redundancy or uniformity present in an image to compress it. This novel method estimates the detail present in a block using Canny's edge detector. Each block is classified into edge and nonedge using a criterion. By merging consecutive nonedge blocks, the block size is varied.  
## Getting Started
### Prerequisites
1. **Anaconda for Python 2.7**  
2. **OpenCV for Python 2.7** 
### Installing
1. **Anaconda for Python 2.7**  
Go to the [downloads page of Anaconda](https://www.anaconda.com/download/) and select the installer for Python 2.7. Once downloaded, installing it should be a straightforward process. Anaconda has along with it most of the packages we need.  

2. **OpenCV for Python 2.7**   
This [page](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html) explains it quite well.   
## Running
Before running `vbs.py`, there are a few parameters to be set.  
```python
image_location
bits_per_codevector
block_width
block_height
mu
threshold_1
threshold_2
```  
`image_location` is set to the relative location of the image from the current directory.  
`bits_per_codevector` is set based on the size of the codebook you desire. For e.g., for a 256-vector codebook, this value should be `8` as `2^8 = 256`.  
`block_width` and `block_height` are set to the size of the blocks the image is divided into. Make sure the blocks cover the the entire image.  
`mu` is the coefficient used in the block-classifying criteria.  
`threshold_1` and `threshold_2` are the [parameters](https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de) used by Canny edge detector.  
`perturbation_vector` in `lbg_split.py` can be changed manually.  

Once the parameters are set, enter the following command to run the script.  
`python [name of the script] [image_location] [bits_per_codevector] [block_width] [block_height] [mu] [threshold_1] [threshold_2]`  

