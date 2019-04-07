import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import sys
import argparse
import time
from PIL import Image
import cv2
import csv
import os

from ocr_dataset import OcrDataset
import torch
from torch.utils.data import DataLoader
from ocr_dataset_union import OcrDatasetUnion
from datautils import GroupedSampler, SortByWidthCollater
import imagetransforms
import augment

'''
2018-06-20

PyTorch augment test

'''

def parse_arguments(argv):
    parser = argparse.ArgumentParser()


    parser.add_argument("--datadir", type=str, action='append', required=True, help="specify the location to data.")
    parser.add_argument(
        '--output', type=str, help='output directory of augment images',
        default='/exp/YOMDLE/final_arabic/augment')
        
    return parser.parse_args(argv)


    
def main(args):
    start_time = time.clock()

    print(args.datadir)
    #print(args.datadir[0])
    
    line_img_transforms = []
    line_img_transforms.append(imagetransforms.Scale(new_h=30))
    line_img_transforms.append(augment.ImageAug())
    line_img_transforms.append(imagetransforms.ToTensor())
    
    line_img_transforms = imagetransforms.Compose(line_img_transforms)    

    valid_dataset = OcrDataset(args.datadir[0], "validation", line_img_transforms)


    valid_dataloader = DataLoader(valid_dataset,10, num_workers=0,sampler=GroupedSampler(valid_dataset, rand=False),
                                       collate_fn=SortByWidthCollater, pin_memory=False, drop_last=False)
    
    epoch_size = len(valid_dataloader)
    print('Epoch size %d' % (epoch_size))
    
    batch_ctr = 0
    for input_tensor, target, input_widths, target_widths, metadata in valid_dataloader:
        batch_ctr += 1
        image_list = input_tensor.numpy()   
        print(image_list.shape)
        print(len(image_list))
        print(len(image_list[:,0]))
        #print('2', len(image_list[:,0]))
        for ctr in range(0, len(image_list[:,0])):
            print(image_list[ctr,:].shape)
            img = np.uint8(image_list[ctr,:].squeeze().transpose((1,2,0)) * 255)
            
            #img = img.astype(int)
            
            #print(batch_ctr, input_widths.numpy(), image_list.shape, img.shape)
            
            if ctr <= 10:
                cv2.imwrite(args.output + '/' + str(batch_ctr) + '-' + str(ctr) + "_aug.png", img)
                
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #im_pil = Image.fromarray(img)
                #im_pil.save(args.output + '/' + str(batch_ctr) + "-b.png" , 'PNG', quality=100)
            else:
                break
            
    print('...complete %s' % (time.clock() - start_time))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
