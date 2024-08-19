**General Multi-label Image Classification with Transformers**<br/>
Jack Lanchantin, Tianlu Wang, Vicente Ordóñez Román, Yanjun Qi<br/>
Conference on Computer Vision and Pattern Recognition (CVPR) 2021<br/>
[[paper]](https://arxiv.org/abs/2011.14027) [[poster]](https://github.com/QData/C-Tran/blob/main/supplemental/ctran_poster.pdf) [[slides]](https://github.com/QData/C-Tran/blob/main/supplemental/ctran_slides.pdf)
<br/>

## Extension of the C-Tran CVPR repository
**Support and fixes include**<br/>
* Additional Dataset
* Single Image Fwd pass
* Torch upgrade changes


## Submission

The submission features the training code adaptation in this repository and 2 models inside the results folder.
The models are trained for Label Classification Scoring and Label Visibility ratio Scoring respectively.

## Models
### 1. HOK4K
This model is trained to classify the presence of a label. As such its outputs are distributed in the boundaries of the [0-1] probability region.

### 2. HOK4KVIS
This experimental model is trained to classify the presence of a label but also regress its output to the visibility of the region. As such its outputs are more in line with the requested probabilities.

## Results
The file ```car_imgs_4000_results.csv``` shows the output of the two models and their perceived behaviour on the dataset.

A second observation is also made regarding errors in classification of the 'backdoor_left' for images of the right_hand_side of the car.
This can be attributed to the image augmentations from the horizontal flips, which affect side dependant label. It is seen in other applications like human hand detection.

Possible solutions include, having associative labels designed to flip with the image transforms. Disabling the image_transforms.


## Code Execution
### Train
```shell
python main.py  --batch_size 8  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'hok4kvis' --use_lmt --dataroot /data --max_samples -1
```
* Ensure ```CodingChallenge_v2``` is available under the 'data_root' folder
* Params
  * dataset: hok4k, generates the standard classifier model
  * dataset: hok4kvis, generates the second model

### Inference
```shell
python main.py  --batch_size 8  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'hok4ktest' --use_lmt --dataroot /data --max_samples -1 --inference --saved_model_name results/hok4k.3layer.bsz_8.adam1e-05.lmt.unk_loss/best_model.pt
```
* Params
  * inference: runs the standard inference of the C-Tran on test/val subset and outputs metrics
    * use with --dataset 'hok4ktest' to force test on all 4k images
    * use --saved_model_name to model checkpoints file of choice
    * also generates a best_model.csv for all files without shuffling in corresponding directory
  * custom_infer: runs a single image in the forward pass and prints the probabilities to console
    * use --saved_model_name and --image_path respectively