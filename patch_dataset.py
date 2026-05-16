import torch
import random
import os
def check_script():
    with open("scripts/train.py", "r") as f:
        print(f.read()[:50])
check_script()
