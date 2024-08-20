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

The submission features the training code adaptation in this repository and 3 models inside the results folder.
The models are trained for Label Classification Scoring and Label Visibility ratio Scoring respectively. The latter also contains a modified model trained without horizontal flip augmentations.

## Models
### 1. HOK4K
This model is trained to classify the presence of a label. As such its outputs are distributed in the boundaries of the [0-1] probability region.

### 2. HOK4KVIS
This experimental model is trained to classify the presence of a label but also regress its output to the visibility of the region. As such its outputs are more in line with the requested probabilities.

### 3. HOK4KVIS_noflip
This model is trained similar to the previous model but without horizontal flips to better classify side labels.

## Results
The file ```car_imgs_4000_results.csv``` shows the output of the two models and their perceived behaviour on the dataset.

A second observation is also made regarding errors in classification of the 'backdoor_left' for images of the right_hand_side of the car.
This can be attributed to the image augmentations from the horizontal flips, which affect side dependant label. It is seen in other applications like human hand detection.

Possible solutions include, having associative labels designed to flip with the image transforms. Disabling the image_transforms.

A model with the image transforms disabled is also added, which performs better in cases of right side doors as well.


## Code Execution
> The original requirements were designed for torch 1 and Cuda 11. An updated ```req_38_trch2_cu12.txt``` is added for this version. The code has also been updated to work with the suggested libraries.
### Train
```shell
python main.py  --batch_size 8  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'hok4kvis' --use_lmt --dataroot /data --max_samples -1
```
* Ensure ```CodingChallenge_v2``` is available under the 'data_root' folder
* Params
  * dataset: hok4k, generates the standard classifier model
  * dataset: hok4kvis, generates the second model
* Execute ```python create_models.py``` to create the 2 models as dicussed + an experimental model ignoring horizontal flips

### Inference
```shell
python main.py  --batch_size 8  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'hok4ktest' --use_lmt --dataroot /data --max_samples -1 --inference --saved_model_name results/hok4k.3layer.bsz_8.adam1e-05.lmt.unk_loss/best_model.pt
python main.py  --batch_size 8  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'hok4ktest' --use_lmt --dataroot /data --max_samples -1 --infer_custom --saved_model_name results/hok4kvis.3layer.bsz_8.adam1e-05.lmt.unk_loss.no_flip/best_model.pt --image_path /data/CodingChallenge_v2/imgs/0a1d0d53-eaa4-4f42-9ea7-2197bd183520.jpg
```
* Params
  * inference: runs the standard inference of the C-Tran on test/val subset and outputs metrics
    * use with --dataset 'hok4ktest' to force test on all 4k images
    * use --saved_model_name to model checkpoints file of choice
    * also generates a best_model.csv for all files without shuffling in corresponding directory
  * custom_infer: runs a single image in the forward pass and prints the probabilities to console
    * use --saved_model_name and --image_path respectively
* Models
  * Pretrained models as part of the submission aree available under [Models](https://drive.google.com/drive/folders/1mliQB7q0op_6iLP4l6cvoNmIcF7H9xtf?usp=sharing)
  * Download the models folders and link the respective model to test in the param ```--saved_model_name```