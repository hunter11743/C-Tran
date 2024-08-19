import os
cms = [
    r"python main.py  --batch_size 8  --lr 0.00001 --epochs 100 --optim 'adam' --layers 3  --dataset 'hok4kvis' --use_lmt --dataroot /data --max_samples -1 --ignore_flip",
    r"python main.py  --batch_size 8  --lr 0.00001 --epochs 100 --optim 'adam' --layers 3  --dataset 'hok4kvis' --use_lmt --dataroot /data --max_samples -1",
    r"python main.py  --batch_size 8  --lr 0.00001 --epochs 100 --optim 'adam' --layers 3  --dataset 'hok4k' --use_lmt --dataroot /data --max_samples -1",
]

for cm in cms:
    os.system(cm)