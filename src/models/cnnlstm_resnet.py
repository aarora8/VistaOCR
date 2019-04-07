import logging
import math
import time
import os
import sys
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor


import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from textutils import uxxxx_to_utf8

from torch.nn.utils.rnn import pack_padded_sequence as rnn_pack
from torch.nn.utils.rnn import pad_packed_sequence as rnn_unpack

logger = logging.getLogger('root')

def log_add(logx, logy):
    if (logy > logx):
        temp = logx
        logx = logy
        logy = temp

    negdiff = logy - logx

    if negdiff < -20:
        return logx

    return logx + np.log(1.0 + np.exp(negdiff))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    

class CnnOcrModel(nn.Module):
    def get_hyper_params(self):
        return self.hyper_params

    @classmethod
    def FromSavedWeights(cls, weight_file, verbose=True, gpu=None):
        weights = torch.load(weight_file)

        if verbose:
            logger.info("Loading model from: %s" % weight_file)
            logger.info("\tFrom iteration: %d" % weights['iteration'])
            #logger.info("\tWithout LM: Val CER: %.2f\tWER: %.2f" % (100 * weights['val_cer'], 100 * weights['val_wer']))
            logger.info("\tModel Hyperparams = %s" % str(weights['model_hyper_params']))
            logger.info("")

        hp = weights['model_hyper_params']

        # Need to over-ride GPU parameter
        if not gpu is None:
            hp['gpu'] = gpu
        hp['verbose'] = verbose

        # Create model with right hyperparametrs
        model = cls(**hp)

        if 'rtl' in weights:
            model.rtl = weights['rtl']
        else:
            # Default to RTL temporarily
            model.rtl = True

        # Now load weights from previously trained model
        model.load_state_dict(weights['state_dict'], strict=False)

        return model

    def __init__(self, *args, **kwargs):
        super(CnnOcrModel, self).__init__()

        print('init')
        if len(args) > 0:
            raise Exception("Only keyword arguments allowed in CnnOcrModel")

        self.hyper_params = kwargs.copy()

        self.input_line_height = kwargs['input_line_height']
        self.rds_line_height = kwargs['rds_line_height']
        self.alphabet = kwargs['alphabet']
        self.lstm_input_dim = kwargs['lstm_input_dim']
        self.num_lstm_layers = kwargs['num_lstm_layers']
        self.num_lstm_hidden_units = kwargs['num_lstm_hidden_units']
        self.p_lstm_dropout = kwargs['p_lstm_dropout']
        self.num_in_channels = kwargs.get('num_in_channels', 1)

        self.gpu = kwargs.get('gpu', True)
        self.multigpu = kwargs.get('multigpu', True)
        self.verbose = kwargs.get('verbose', True)

        self.lattice_decoder = None

        # Sanity checks
        if self.rds_line_height > self.input_line_height:
            raise Exception("rapid-downsample line height must be less than or equal to input line height")
        if self.input_line_height % self.rds_line_height != 0:
            raise Exception("rapid-downsample line height must evenly divide input line height by a power of 2")

        num_rds_pooling_layers = 0
        lh = self.input_line_height
        while lh > self.rds_line_height:
            num_rds_pooling_layers += 1

            if lh % 2 != 0:
                raise Exception("rapid-downsample line height must eenly diide input line height by a power of 2")
            lh /= 2

        if lh != self.rds_line_height:
            raise Exception("rapid-downsample line height must eenly diide input line height by a power of 2")

        self.rapid_ds = nn.Sequential()

        last_num_filters = self.num_in_channels
        for i in range(num_rds_pooling_layers):
            self.rapid_ds.add_module("%02d-conv"%i,nn.Conv2d(last_num_filters, 16, kernel_size=3, padding=1))
            self.rapid_ds.add_module("%02d-relu"%i,nn.ReLU(inplace=True))
            self.rapid_ds.add_module("%02d-pool"%i,nn.MaxPool2d(2, stride=2))
            last_num_filters = 16

        print('CNN input last_num_filters %d' % (last_num_filters))
#         self.cnn = nn.Sequential(
#             *self.ConvBNReLU(last_num_filters, 64),
#             *self.ConvBNReLU(64, 64),
#             nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.7)),
#             *self.ConvBNReLU(64, 128),
#             *self.ConvBNReLU(128, 128),
#             nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.7)),
#             *self.ConvBNReLU(128, 256),
#             *self.ConvBNReLU(256, 256),
#             *self.ConvBNReLU(256, 256)
#         )


#         self.in_planes = int(16)
#         #depth = int(28)
#         depth = int(10)
#         #depth = int(16)
#         widen_factor = int(10) 
#         dropout_rate = 0.3
#         
#         assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
#         n = int((depth-4)/6)
#         k = widen_factor
# 
#         print('| Wide-Resnet %dx%d' %(depth, k))
#         nStages = [16, 16*k, 32*k, 64*k]
# 
#         print('...Stages', nStages)
#         print('...depth %d' % (depth))
#         print('...widen_factor %d' % (widen_factor))
#         print('...dropout_rate %f' % (dropout_rate))
#
#         self.cnn = nn.Sequential(
#             conv3x3(3,nStages[0]),
#             self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1),
#             self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2),
#             self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2),
#             nn.BatchNorm2d(nStages[3], momentum=0.9)
#         )
 

        block = BasicBlock
        layers  = [2, 2, 2, 2]
        zero_init_residual=False
        self.inplanes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.FractionalMaxPool2d(2, output_ratio=(0.7, 0.7)),
            self._make_layer(block, 64, layers[0]),
            nn.FractionalMaxPool2d(2, output_ratio=(0.7, 0.7)),
            self._make_layer(block, 128, layers[1], stride=1),
            nn.FractionalMaxPool2d(2, output_ratio=(0.7, 0.7)),
            self._make_layer(block, 256, layers[2], stride=1),
            nn.FractionalMaxPool2d(2, output_ratio=(0.7, 0.7)),
            self._make_layer(block, 512, layers[3], stride=1)
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                            
        
        # We need to calculate cnn output size to construct the bridge layer
        fake_input_width = 500
        cnn_out_h, cnn_out_w = self.cnn_input_size_to_output_size((self.input_line_height, fake_input_width))
        print('CNN out height %d, width %d' % (cnn_out_h, cnn_out_w))
        cnn_out_c = self.cnn_output_num_channels()

        #cnn_out_h = 4
        cnn_feat_size = cnn_out_c * cnn_out_h

        print('CNN out channels %d' % (cnn_out_c))
        print('CNN feature size %d' % (cnn_feat_size))


        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.lstm_input_dim),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTM(self.lstm_input_dim, self.num_lstm_hidden_units, num_layers=self.num_lstm_layers,
                            dropout=self.p_lstm_dropout, bidirectional=True)


        self.prob_layer = nn.Sequential(
            nn.Linear(2 * self.num_lstm_hidden_units, len(self.alphabet))
        )


        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

        total_params = 0
        for param in self.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            total_params += local_params

        cnn_params = 0
        for param in self.cnn.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            cnn_params += local_params

        lstm_params = 0
        for param in self.lstm.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            lstm_params += local_params

        if self.verbose:
            logger.info("Total Model Params = %d" % total_params)
            logger.info("\tCNN Params = %d" % cnn_params)
            logger.info("\tLSTM Params = %d" % lstm_params)

            logger.info("Model looks like:")
            logger.info(repr(self))


        if torch.cuda.device_count() > 0:   
            print('...available %d GPUs' % torch.cuda.device_count())  
        else:
            print('...available CPU')
            
        if torch.cuda.is_available() and self.gpu:
            print('...using GPUS')
            self.rapid_ds = self.rapid_ds.cuda()
            self.cnn = self.cnn.cuda()
            self.bridge_layer = self.bridge_layer.cuda()
            self.lstm = self.lstm.cuda()
            self.prob_layer = self.prob_layer.cuda()

            if self.multigpu:
                print('...using MULTIPLE GPUS')
                self.cnn = torch.nn.DataParallel(self.cnn)
        else:
            logger.info("Warning: Runnig model on CPU")


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    def cnn_output_num_channels(self):
        out_c = 0
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
        return out_c


    def calculate_hw(self, module, out_h, out_w):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d):
            
            if isinstance(module.padding, tuple):
                padding_y, padding_x = module.padding
            else:
                padding_y = padding_x = module.padding
            if isinstance(module.dilation, tuple):
                dilation_y, dilation_x = module.dilation
            else:
                dilation_y = dilation_x = module.dilation
            if isinstance(module.stride, tuple):
                stride_y, stride_x = module.stride
            else:
                stride_y = stride_x = module.stride
            if isinstance(module.kernel_size, tuple):
                kernel_size_y, kernel_size_x = module.kernel_size
            else:
                kernel_size_y = kernel_size_x = module.kernel_size

            out_h = math.floor((out_h + 2.0 * padding_y - dilation_y * (kernel_size_y - 1) - 1) / stride_y + 1)
            out_w = math.floor((out_w + 2.0 * padding_x - dilation_x * (kernel_size_x - 1) - 1) / stride_x + 1)
            
        elif isinstance(module, nn.FractionalMaxPool2d):
            if module.output_size is not None:
                out_h, out_w = module.output_size
            else:
                rh, rw = module.output_ratio
                out_h, out_w = math.floor(out_h * rh), math.floor(out_w * rw)


#            if module.outh is not None:
#                #out_h, out_w = module.output_size
#                out_h, out_w = module.outh, module.outw
#            else:
#                #rh, rw = module.output_ratio
#                rh, rw = module.rh, module.rw
#                out_h, out_w = math.floor(out_h * rh), math.floor(out_w * rw)

        return out_h, out_w

    def cnn_input_size_to_output_size(self, in_size):
        out_h, out_w = in_size


        # First do rapid downsampling
        #for module in self.rapid_ds.modules():
        #    out_h, out_w = self.calculate_hw(module, out_h, out_w)
        #print('----start cnn_input_size_to_output_size')
        for module in self.cnn.modules():
            #print(module.__class__.__name__, out_h, out_w)
            out_h, out_w = self.calculate_hw(module, out_h, out_w)
        #print('END')
        
        #out_h = math.floor(out_h * 0.7 * 0.7 * 0.7 - 1)
        out_h = 6
        #out_w = math.floor(out_w * 0.8 * 0.8 * 0.8)
        
        return (out_h, out_w)

    # Consider: nn.Dropout2d
    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]

    def forward(self, x, actual_minibatch_widths, summary_flag=False):
        #print('--->Forward CnnOcrModel')

        #print('x ', x.size())
        rapid_ds_output = self.rapid_ds(x)
        #print('rapid_ds_output', rapid_ds_output.size())
        cnn_output = self.cnn(rapid_ds_output)
        cnn_output = F.relu(cnn_output)
        
        #cnn_output = F.avg_pool2d(out, 8)
        #out = out.view(out.size(0), -1)
        
#         out = self.conv1(rapid_ds_output)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         cnn_output = F.relu(self.bn1(out))
        #cnn_output = F.avg_pool2d(out, 8)
        #cnn_output = out.view(out.size(0), -1)
        
        
        b, c, h, w = cnn_output.size()
        #print('output size b %d, c %d, h %d, w %d' % (b, c, h, w))
        cnn_output = cnn_output.permute(3, 0, 1, 2).contiguous()

        if summary_flag == True:
            return cnn_output
 
 
        lstm_input = self.bridge_layer(cnn_output.view(-1, c * h)).view(w, b, -1)
        #print('lstm input shape', lstm_input.size())

        #if summary_flag == True:
        #    return lstm_input

        # Try tensor.unfold(0, frame_size, step_size), e.g. with frame_size=2, step_size=1
 
        # Note: pack_padded_sequence assumes that minibatch elements are sorted by length
        #       i.e. minibatch_widths[0] is longest sequence and minibatch_widths[-1] is shortest sequence
        #       We assume that the data loader arranged input to conform to this constraint
 
        
        actual_cnn_output_widths = [self.cnn_input_size_to_output_size((self.input_line_height, width))[1] for width in
                                        actual_minibatch_widths.data]
        #print('...actual_cnn_output_widths', actual_cnn_output_widths)
 
        packed_lstm_input = rnn_pack(lstm_input, actual_cnn_output_widths)
        packed_lstm_output, _ = self.lstm(packed_lstm_input)
        lstm_output, lstm_output_lengths = rnn_unpack(packed_lstm_output)
 
        w = lstm_output.size(0)
 
        prob_output = self.prob_layer(lstm_output.view(-1, lstm_output.size(2))).view(w, b, -1)

 
        return prob_output, lstm_output_lengths.to(torch.int32)
        #return lstm_output, lstm_output_lengths.to(torch.int32)

    def init_lm(self, lm_file, word_sym_file, lm_units, acoustic_weight=0.8, max_active=5000, beam=16.0, lattice_beam=10.0):
        # Only pull in if needed
        script_path = os.path.dirname(os.path.realpath(__file__))
        #sys.path.append(script_path + "/../eesen")
        sys.path.insert(0,'/home/hltcoe/srawls/pyeesen')
        import eesen
        print("Loading eesen from: %s" % eesen.__file__)
        self.acoustic_weight = acoustic_weight
        self.lattice_decoder = eesen.LatticeDecoder(lm_file, word_sym_file, self.acoustic_weight, max_active, beam,
                                                    lattice_beam)

        # Need to keep track of model-alphabet to LM-alphabet conversion
        units = ['<ctc-blank>']
        with open(lm_units, 'r') as fh:
            for line in fh:
                units.append(line.strip().split(' ')[0])

        self.lmidx_to_char = units
        self.lmchar_to_idx = dict(zip(units, range(len(units))))
        

        # Let's precompute some stuff to make lm faster
        print("Prep work...")
        self.lm_swap_idxs = []
        self.lm_swap_idxs_modelidx = []
        self.lm_swap_idxs_lmidx = []
        self.add_to_blank_char = []
        self.add_to_blank_idx = []
        for model_idx in range(len(self.alphabet.idx_to_char)):
            char = self.alphabet.idx_to_char[model_idx]
            if not char in self.lmchar_to_idx:
                self.add_to_blank_char.append(char)
                self.add_to_blank_idx.append(model_idx)
                continue
            lm_idx = self.lmchar_to_idx[char]
            self.lm_swap_idxs.append( (model_idx,lm_idx) )
            self.lm_swap_idxs_modelidx.append(model_idx)
            self.lm_swap_idxs_lmidx.append(lm_idx)
        print("Done prep work")
        print("\tFYI: these chars were in model but not in LM:  %s" % str(self.add_to_blank_char))



    def decode_with_lm_mt(self, model_output, batch_actual_timesteps, uxxxx=False, n_workers=10):
        # Setup multi-threaded decoding

        #print("About to create threadpool")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:

            if self.lattice_decoder is None:
                raise Exception("Must initialize lattice decoder prior to LM decoding")

            T = model_output.size()[0]
            B = model_output.size()[1]

            # Actual model output is not set to probability vector yet, need to run softmax
            probs = torch.nn.functional.log_softmax(model_output.view(-1, model_output.size(2)), dim=1).view(model_output.size(0),
                                                                                                      model_output.size(1),
                                                                                                      -1)

            # Need to take care of issue where prob goes to a char in model-alphabet but not in lm-alphabet
            # Just assign high prob to ctc-blank?
            #print("Sum of missing chars' prob = %s" % str(model_output[:,:,self.add_to_blank_idx].sum(dim=2)))
            #probs[:,:,0] += probs[:,:,self.add_to_blank_idx].sum(dim=2)
            #probs[:,:,self.add_to_blank_idx] = 0

            # Make sure we're on CPU
            probs = probs.data.cpu()

            # We process decoder parallely in worker threads; store those async futures here
            decoder_futures = [None]*B

            # probs = probs * self.acoustic_weight
            start_submitting = datetime.now()
            for b in range(B):
                probs_remapped = np.full( (batch_actual_timesteps[b], len(self.lmidx_to_char)), np.log(1e-10))
                probs_remapped[:,self.lm_swap_idxs_lmidx] = probs[:batch_actual_timesteps[b], b, self.lm_swap_idxs_modelidx]

                decoder_futures[b] = executor.submit(self.lattice_decoder.Decode, probs_remapped)

            end_submitting = datetime.now()
            #print("Waiting for threadpool jobs to finish. Took %f s to get here" % (end_submitting - start_submitting).total_seconds())
        # At this point all decoder tasks are done (we are outside scope of with ThreadPoolExecutor, so it has finished)
        end_waiting = datetime.now()
        #print("Took %f s to wait for batch decodes to finish" % (end_waiting - end_submitting).total_seconds())

        hyp_results = []

        for b in range(B):
            res = decoder_futures[b].result()

            res_utf8 = ''
            if uxxxx == False:
                for uxxxx_word in res.split(' '):
                    res_utf8 += ''.join([uxxxx_to_utf8(r) for r in uxxxx_word.split('_')])
                res = res_utf8
            else:
                res_flatten = ''
                for uxxxx_word in res.split(' '):
                    for uxxxx_char in uxxxx_word.split('_'):
                        res_flatten += uxxxx_char
                        res_flatten += ' '
                res = res_flatten.strip()

            hyp_results.append(res)

        return hyp_results




    def decode_with_lm(self, model_output, batch_actual_timesteps, uxxxx=False, pmod=False):

        if self.lattice_decoder is None:
            raise Exception("Must initialize lattice decoder prior to LM decoding")

        T = model_output.size()[0]
        B = model_output.size()[1]

        # Actual model output is not set to probability vector yet, need to run softmax
        probs = torch.nn.functional.log_softmax(model_output.view(-1, model_output.size(2)), dim=1).view(model_output.size(0),
                                                                                                  model_output.size(1),
                                                                                                  -1)

        # Make sure we're on CPU
        probs = probs.data.cpu()

        hyp_results = []
        for b in range(B):

            for t in range(T):
                if pmod:
                    if torch.max(probs[t,b]) < 0.9:
                        probs[t,b] = probs[t,b] * 0.6  # low conf
                    else:
                        probs[t,b] = probs[t,b] * 1.1 # 'normal, high conf'
                else:
                    probs[t,b] = probs[t,b] * self.acoustic_weight


            activations = probs[:, b, :].numpy()
            activations_remapped = np.zeros((batch_actual_timesteps[b], len(self.lmidx_to_char)))

            for c in range(len(self.lmidx_to_char)):
                char = self.lmidx_to_char[c]
                if char in self.alphabet.char_to_idx:
                    mapped_c = self.alphabet.char_to_idx[char]
                    activations_remapped[:, c] = activations[:batch_actual_timesteps[b], mapped_c]
                else:
                    activations_remapped[:, c] = np.log(1e-10)

            # Now check that anything turned to NULL gets mapped to ctc-blank
            for t in range(batch_actual_timesteps[b]):
                psum = np.log(1e-10)
                for c in range(len(self.lmidx_to_char)):
                    psum = np.logaddexp(psum, activations_remapped[t, c])
                if psum < np.log(1e-2):
                    activations_remapped[t, 0] = 0



            res = self.lattice_decoder.Decode(activations_remapped)
            res_utf8 = ''
            if uxxxx == False:
                for uxxxx_word in res.split(' '):
                    res_utf8 += ''.join([uxxxx_to_utf8(r) for r in uxxxx_word.split('_')])
                res = res_utf8
            else:
                res_flatten = ''
                for uxxxx_word in res.split(' '):
                    for uxxxx_char in uxxxx_word.split('_'):
                        res_flatten += uxxxx_char
                        res_flatten += ' '
                res = res_flatten.strip()

            hyp_results.append(res)

        return hyp_results



    def decode_without_lm(self, model_output, batch_actual_timesteps, uxxxx=False):
        start_decode = datetime.now()
        min_prob_thresh = 3 * 1 / len(self.alphabet)

        T = model_output.size()[0]
        B = model_output.size()[1]

        prev_char = ['' for _ in range(B)]
        result = ['' for _ in range(B)]

        for t in range(T):

            # #gpu argmax (bug!!!!!)
            # gpu_argmax = True
            # argmaxs, argmax_idxs = model_output.data[t].max(dim=1)
            # argmaxs.squeeze_()
            # argmax_idxs.squeeze_()    

            # cpu argmax
            gpu_argmax = False
            model_output_at_t_cpu = model_output.data[t].cpu().numpy()
            argmaxs = model_output_at_t_cpu.max(1).flatten()
            argmax_idxs = model_output_at_t_cpu.argmax(1).flatten()

            for b in range(B):
                # Only look at valid model output for this batch entry
                if t >= batch_actual_timesteps[b]:
                    continue

                if argmax_idxs[b] == 0:  # CTC Blank
                    prev_char[b] = ''
                    continue

                # Heuristic
                # If model is predicting very low probability for all letters in alphabet, treat that the
                # samed as a CTC blank
                if argmaxs[b] < min_prob_thresh:
                    prev_char[b] = ''
                    continue


                char = self.alphabet.idx_to_char[argmax_idxs[b]]

                if prev_char[b] == char:
                    continue

                result[b] += char
                prev_char[b] = char

                # Add a space to all but last iteration
                if t != T - 1:
                    result[b] += ' '

        # Strip off final token-stream space if needed
        for b in range(B):
            if len(result[b]) > 0 and result[b][-1] == ' ':
                result[b] = result[b][:-1]

        # Check if we should return utf8 output
        if uxxxx == False:
            result = [uxxxx_to_utf8(r) for r in result]

        return result
