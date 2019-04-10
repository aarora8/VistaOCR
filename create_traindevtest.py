#!/usr/bin/env python3

import os
import numpy as np
import cv2
import json
import lmdb
import csv
import math
import sys

sys.path.append("./src")
import textutils


gt_basedir = '/export/b04/aarora8/aavista2/data/download/russian/truth_csv'
image_basedir = '/export/b04/aarora8/aavista2/data/download/russian/truth_line_image/'
output_dir = '/export/b04/aarora8/aavista2/data/lmdb/'


def main():
    page_csvs = np.array(os.listdir(gt_basedir))
    train_pages, dev_pages, test_pages = get_train_dev_test_split(page_csvs)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_json = {'train': [], 'test': [], 'validation': []}
    lmdb_env = lmdb.Environment(os.path.join(output_dir, 'line-images.lmdb'), map_size=1e12)
    
    with lmdb_env.begin(write=True) as txn:
        print("Processing Training")
        process_set(train_pages, out_json['train'], txn)

        print("Processing Validation")
        process_set(dev_pages, out_json['validation'], txn)

        print("Processing Test")
        process_set(test_pages, out_json['test'], txn)
        
    with open(os.path.join(output_dir, 'desc.json'), 'w', encoding='utf8') as fh:
        json.dump(out_json, fh)

    print("Done.")


def process_set(pages, out_json_ary, txn):
    total_line_cnt = 0
    for page in pages:
        for line_num, (img_bn, text) in enumerate(load_gt_csv(page)):
            total_line_cnt += 1
            if total_line_cnt % 500 == 0:
                print("Processed %d lines" % total_line_cnt)

            # Okay now crop out line image
            line_img = cv2.imread(os.path.join(image_basedir, img_bn))

            if line_img is None:
                print("Couldn't read img: %s" % os.path.join(image_basedir, img_bn))
                continue
            
            key = img_bn.replace(".png", "_") + str(line_num)
            retval, img_bytes = cv2.imencode(".png", line_img)


            # Make sure conversion was okay
            assert retval
            txn.put(key.encode("ascii"), img_bytes)

            uxxxx_trans = textutils.utf8_to_uxxxx(text)

            if len(uxxxx_trans) == 0:
                print("key = %s" % key)
                continue

            out_json_ary.append( {'id': key, 'trans': uxxxx_trans, 'height': line_img.shape[0], 'width': line_img.shape[1]} )



def load_gt_csv(csv_file):
    page_img_bn = csv_file.replace(".csv",".png")
    
    data = []
    with open(os.path.join(gt_basedir, csv_file), 'r', encoding='utf8') as fh:
        csv_reader = csv.reader(fh)

        # Skip header
        csv_reader.__next__()

        for row in csv_reader:
            uid,line_img_bn,x1,y1,x2,y2,x3,y3,x4,y4,confidence,text = row[:12]

            # Sanity check
            page_img_bn_ = line_img_bn[ :line_img_bn.rfind('_')] + line_img_bn[ line_img_bn.rfind('.') :]
            assert page_img_bn_ == page_img_bn

            data.append( (line_img_bn, text) )

    return data



def get_train_dev_test_split(page_images):
    ntotal = len(page_images)
    ntrain = math.floor(0.8*ntotal)
    ndev = math.floor(0.1*ntotal)
    ntest = ntotal - ntrain - ndev

    random_permuation = np.random.permutation(ntotal)
    train_idxs = random_permuation[:ntrain]
    dev_idxs = random_permuation[ntrain:ntrain+ndev]
    test_idxs = random_permuation[ntrain+ndev:]

    # Sanity check
    assert len(train_idxs) == ntrain
    assert len(dev_idxs) == ndev
    assert len(test_idxs) == ntest
    
    train_pages = page_images[train_idxs]
    dev_pages = page_images[dev_idxs]
    test_pages = page_images[test_idxs]

    return train_pages, dev_pages, test_pages


if __name__ == '__main__':
    main()
