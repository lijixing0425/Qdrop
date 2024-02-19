import os
import argparse
import time

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("exp_name", type=str,
    #                     choices=['resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    # args = parser.parse_args()
    w_bits = [2, 4, 3, 4]
    a_bits = [2, 2, 3, 4]
    # w_bits = [2]
    # a_bits = [2]
    model_names = ['resnet18', 'regnetx_600m', 'resnet50', 'mobilenetv2', 'regnetx_3200m', 'mnasnet']
    for model_name in model_names:
        print('begin run {}, lijx............................lijx'.format(model_name))
        if model_name == "resnet18":
            for i in range(4):
                os.system(
                    f"python main_imagenet.py  --arch resnet18 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01")
                time.sleep(0.5)

        if model_name == "resnet50":
            for i in range(4):
                os.system(
                    f"python main_imagenet.py  --arch resnet50 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01")
                time.sleep(0.5)

        if model_name == "regnetx_600m":
            for i in range(4):
                os.system(
                    f"python main_imagenet.py  --arch regnetx_600m --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01")
                time.sleep(0.5)

        if model_name == "regnetx_3200m":
            for i in range(4):
                os.system(
                    f"python main_imagenet.py  --arch regnetx_3200m --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.01")
                time.sleep(0.5)

        if model_name == "mobilenetv2":
            for i in range(4):
                os.system(
                    f"python main_imagenet.py  --arch mobilenetv2 --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.1")
                time.sleep(0.5)

        if model_name == "mnasnet":
            for i in range(4):
                os.system(
                    f"python main_imagenet.py  --arch mnasnet --n_bits_w {w_bits[i]} --n_bits_a {a_bits[i]} --weight 0.2")
                time.sleep(0.5)

