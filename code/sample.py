import sys
import os
import sys
import argparse
import shutil
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEN12MS_dir', type=str, default=None)
    parser.add_argument('--sample_output_dir', type=str, default=None)

    args = parser.parse_args()
    assert os.path.exists(args.SEN12MS_dir)
    assert os.path.exists(args.sample_output_dir)
    
    path = os.listdir(args.SEN12MS_dir)
    lc = []
    # source = []
    # target = []
    for folder in path:
        if folder[:2]=="lc":
            lc.append(folder[3:]) 
    lc = np.random.choice(np.array(lc), 5, replace=True)
    
    for folder in path:
        if folder[3:] in lc:
            source = os.path.join(args.SEN12MS_dir, folder)
            target = os.path.join(args.sample_output_dir, folder)
            shutil.copytree(source, target)
main()