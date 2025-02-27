# YOLOv3 🚀 by Ultralytics, GPL-3.0 license

"""
Standalone downloader script for datasets.
Run this to download datasets to a datastore machine without installing
heavy libs like PyTorch.

Usage:
 python downloadhelper.py <DATASET.YAML>

Example:
 python downloadhelper.py data/objects365.yaml

Package Requirements:
- numpy
- tqdm
- pycocotools
- PyYAML
"""

import glob
import logging
import math
import os
import sys
import platform
import random
import re
import shutil
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import numpy as np
import yaml

import imp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory



def main():
    # Because the data YAML scripts import 'utils/general.py' but that
    # script has a lot of dependencies unnecessary for downloading - like
    # torch and cv2 - we need a way to mock the 'utils' and 'utils.general'
    # modules. 
    # This function uses the imp package to create these mock modules
    # dynamically.
    setup_import_stubs()

    datafile = sys.argv[1]
    if datafile.endswith('yaml'):
        check_dataset(datafile)
        
    else:
        if datafile == 'objects365':
            download_objects365()


def setup_import_stubs():
    # Create mock'utils' and 'utils.general' modules and import them
    # dynamically.
    # Based on  https://www.oreilly.com/library/view/python-cookbook/0596001673/ch15s03.html
    
    utils = imp.new_module('utils')
    #print(utils)
    sys.modules['utils'] = utils
    
    utils_general_module = imp.new_module('utils.general')
    #print(utils_general_module)
    sys.modules['utils.general'] = utils_general_module
    utils.general = utils_general_module
    
    utils.general.Path = Path
    utils.general.download = download
    utils.general.np = np
    utils.general.xyxy2xywhn = xyxy2xywhn


def check_dataset(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):  # i.e. gs://bucket/dir/coco128.zip
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Parse yaml
    path = extract_dir or Path(data.get('path') or '')  # optional 'path' default to '.'
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))

    # val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
    paths = [Path(x).resolve() for x in (train, val, test)]  # check paths
    if not all(x.exists() for x in paths):
        print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in paths if not x.exists()])
        if s and autodownload:  # download script
            root = path.parent if 'path' in data else '..'  # unzip directory i.e. '../'
            if s.startswith('http') and s.endswith('.zip'):  # URL
                f = Path(s).name  # filename
                print(f'Downloading {s} to {f}...')
                
                # Remove dependency on Torch:
                #   torch.hub.download_url_to_file(s, f)
                #   Path(root).mkdir(parents=True, exist_ok=True)  # create root
                #   ZipFile(f).extractall(path=root)  # unzip
                #   Path(f).unlink()  # remove zip
                
                download(s, dir=root, unzip=True, delete=True, threads=1)
                    
                r = None  # success
            elif s.startswith('bash '):  # bash script
                print(f'Running {s} ...')
                r = os.system(s)
            else:  # python script
                r = exec(s, {'yaml': data})  # return None
            print(f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
        else:
            raise Exception('Dataset not found.')

    return data  # dictionary



def download(url, dir='.', unzip=True, delete=True, curl=True, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    #
    # The curl arg isn't really required because that's the only way to download implemented,
    # but that arg is necessary for compatibility with code in the data YAML scripts.
    
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f'Downloading {url} to {f}...')
            
            # Remove dependency on torch libs
            #if curl:
            #    os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            #else:
            #    torch.hub.download_url_to_file(url, f, progress=True)  # torch download
            
            os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
            
        if unzip and f.suffix in ('.zip', '.gz'):
            
            if f.suffix == '.zip':
                print(f'Unzipping {f}...')
                ZipFile(f).extractall(path=dir)  # unzip
                
            elif f.suffix == '.gz':
                # Check if the archive's already been unzipped using a number of heuristics:
                # 1. Number of files should match. But this is a problem to implement generically
                #   because we don't know the directory names inside the archive to compare.
                #   For future ref, this can be implemented as follows:
                #      Get number of files (not dirs) in archive: os.popen(f'tar tf {f} -C {f.parent} | grep -e "[^/]$" | wc -l').read()
                #      List only the dirs inside the TAR : tar tvf patch0.tar.gz | grep -e "^[d]"
                #      For each such directory inside the TAR, check if it exists on the filesystem
                #           and get # of files in that directory.
                #
                # 2. If 'tar diff' with unproblematic diffs removed starts producing output immediately, 
                #    something's wrong.
                #
                # 3. If all the above are fine, do a full 'tar diff' while filtering out unproblematic diffs.
                #    If there's no output at all, it's fine.
                
                print(f'Checking if {f} should be unzipped...')
                do_unzip = False

                # TODO There's a mistake here : read() only reads stdout but tar errors are output to stderr. Use subprocess.Popen()
                # instead of os.popen(). Use shlex.split(cmdline) to split the cmdline into args compatible with Popen().
                output = os.popen(f"tar df {f} -C {f.parent} | head -n100 | awk '!/Mode/ && !/Uid/ && !/Gid/ && !/time/'").read()
                if len(output) > 0:
                    # tar diff's immediately started showing errors, possibly "cannot find file" errors. 
                    # To be on the safe size, unzip the archive again.
                    print('tar diff found immediate differences. Will unzip again.')
                    do_unzip = True
                else:
                    # Now try a full df just to be sure.
                    output = os.popen(f"tar df {f} -C {f.parent} | awk '!/Mode/ && !/Uid/ && !/Gid/ && !/time/'").read()
                    if len(output) > 0:
                        # Full diff shows some critical differences. Unzip the archive.
                        print('full tar diff found differences. Will unzip again.')
                        do_unzip = True

                if do_unzip:
                    print(f'Unzipping {f}...')
                    os.system(f'tar xfz {f} -C {f.parent}')  # unzip
                else:
                    print('Archive already unzipped.')

            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    
    # Disable multi-thread downloads because they tend to cause failures
    # in remote SFTP mounts.
    #if threads > 1:
    #    pool = ThreadPool(threads)
    #    pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
    #    pool.close()
    #    pool.join()
    #else:
    for u in [url] if isinstance(url, (str, Path)) else url:
        download_one(u, dir)



def download_objects365():
    # Code copied from data/objects365.yaml
    
    from pycocotools.coco import COCO
    from tqdm import tqdm
    
    # Make Directories
    rootdir = Path(sys.argv[2])  # dataset root dir
    
    for p in 'images', 'labels':
        (rootdir / p).mkdir(parents=True, exist_ok=True)
        for q in 'train', 'val':
            (rootdir / p / q).mkdir(parents=True, exist_ok=True)
    
    # WARNING: This COCO step requires 25GiB RAM to load the JSON and
    # process it. Run it in Colab with a custom GCE highmem VM.
    
    # WARNING 2: This process is slooooow! For each of the 365 classes,
    # it'll iterate through anywhere from 400,000 to 1.38 million JSON entries depending
    # on the classes each image has.
    # Best to keep the JSON on local drives rather than SFTP/remote drives.
    # One it's all completed, transfer just the output label files to remote drive.
    for split in ['train', 'val']:
        images, labels = rootdir / 'images' / split, rootdir / 'labels' / split
    
        coco = COCO(rootdir / f'zhiyuan_objv2_{split}.json')
        names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
        for cid, cat in enumerate(names):
            catIds = coco.getCatIds(catNms=[cat])
            imgIds = coco.getImgIds(catIds=catIds)
            for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
                width, height = im["width"], im["height"]
                path = Path(im["file_name"])  # image filename
                try:
                    label_file = labels / path.with_suffix('.txt').name
                    with open(label_file, 'a') as f:
                        annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
                        for a in coco.loadAnns(annIds):
                            x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                            xyxy = np.array([x, y, x + w, y + h])[None]  # pixels(1,4)
                            x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]  # normalized and clipped
                            #print(f'Writing {label_file}')  # Don't log this. It slows the process down.
                            f.write(f"{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
                except Exception as e:
                    print(e)
    

# Required by download script inside data YAML files.
def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2



if __name__ == '__main__':
    main()

