import os

import torch
from torch import nn

from torch.utils.data import DataLoader
import imagetransforms

import loggy

logger = loggy.setup_custom_logger('root', "train_cnn_lstm.py")

import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import time
import shutil

from ocr_dataset import OcrDataset
from ocr_dataset_union import OcrDatasetUnion
from datautils import GroupedSampler, SortByWidthCollater
from models.cnnlstm import CnnOcrModel
from textutils import *
import argparse
import augment
# import augment_v2
# import augment_v3
# import augment_v4
import cv2

from lr_scheduler import ReduceLROnPlateau

from collections import OrderedDict

def test_on_val(val_dataloader, model, criterion, samples_dir):
    start_val = time.time()
    cer_running_avg = 0
    wer_running_avg = 0
    loss_running_avg = 0
    n_samples = 0

    display_hyp = True

    # To start, put model in eval mode
    model.eval()

    logger.info("About to compute %d val batches" % len(val_dataloader))

    total_val_images = 0
    # No need for backprop during validation test
    with torch.no_grad():
        for input_tensor, target, input_widths, target_widths, metadata in val_dataloader:
            input_tensor = input_tensor.cuda(async=True)

            model_output, model_output_actual_lengths = model(input_tensor, input_widths)
            
            #probs = torch.nn.functional.log_softmax(model_output.view(-1, model_output.size(2)), dim=1).view(model_output.size(0), model_output.size(1),-1)

            loss = criterion(model_output, target, model_output_actual_lengths, target_widths)

            hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=True)

            batch_size = input_tensor.size(0)
            total_val_images += batch_size
            
            ''' CTC CHANGE '''
            #curr_loss = loss.data[0] / batch_size
            curr_loss = loss.item() / batch_size
            
            n_samples += 1
            loss_running_avg += (curr_loss - loss_running_avg) / n_samples

            cur_target_offset = 0
            batch_cer = 0
            batch_wer = 0
            target_np = target.data.numpy()
            ref_transcriptions = []
            for i in range(len(hyp_transcriptions)):
                ref_transcription = form_target_transcription(
                    target_np[cur_target_offset:(cur_target_offset + target_widths.data[i])],
                    model.alphabet
                )
                ref_transcriptions.append(uxxxx_to_utf8(ref_transcription))
                cur_target_offset += target_widths.data[i]
                cer, wer = compute_cer_wer(hyp_transcriptions[i], ref_transcription)

                if cer != None and wer != None:
                    batch_cer += cer
                    batch_wer += wer

            cer_running_avg += (batch_cer / batch_size - cer_running_avg) / n_samples
            wer_running_avg += (batch_wer / batch_size - wer_running_avg) / n_samples

            # For now let's display one set of transcriptions every test, just to see improvements
            if display_hyp:
                logger.info("--------------------")
                logger.info("Sample hypothesis / reference transcripts")
                logger.info("Error rate for this batch is:\tNo LM CER: %f\tWER:%f" % (
                batch_cer / batch_size, batch_wer / batch_size))

                hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=False)
                image_list = input_tensor.cpu().numpy()   
                for i in range(len(hyp_transcriptions)):
                    meta_id = metadata['utt-ids'][i]
                    logger.info("\tHyp[%d]: %s" % (i, hyp_transcriptions[i]))
                    logger.info("\tRef[%d]: %s" % (i, ref_transcriptions[i]))
                    logger.info("\tId[%s]: " % (meta_id))
                    logger.info("")
                    
                    if i < 10:
                        img = np.uint8(image_list[i,:].squeeze().transpose((1,2,0)) * 255)
                        cv2.imwrite(samples_dir + '/' + 'validation_' + str(meta_id) + ".png", img)

                logger.info("--------------------")
                display_hyp = False

    # Finally, put model back in train mode
    model.train()
    end_val = time.time()
    logger.info("Total val time: %s for %d images" % (end_val - start_val, total_val_images))
    return loss_running_avg, cer_running_avg, wer_running_avg

def test_on_val_writeout(val_dataloader, model, out_hyp_path):
    start_val = time.time()

    # To start, put model in eval mode
    model.eval()

    logger.info("About to comptue %d val batches" % len(val_dataloader))

    # No need for backprop during validation test
    with torch.no_grad(), open(out_hyp_path, 'w') as fh_out:
        for input_tensor, target, input_widths, target_widths, metadata in val_dataloader:
            input_tensor = input_tensor.cuda(async=True)

            model_output, model_output_actual_lengths = model(input_tensor, input_widths)
            
            #probs = torch.nn.functional.log_softmax(model_output.view(-1, model_output.size(2)), dim=1).view(model_output.size(0), model_output.size(1),-1)

            hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=False)

            for i in range(len(hyp_transcriptions)):
                hyp_utf8 = utf8_visual_to_logical(hyp_transcriptions[i])
                uttid = metadata['utt-ids'][i]
                fh_out.write("%s (%s)\n" % (hyp_utf8, uttid))


    # Finally, put model back in train mode
    model.train()
    end_val = time.time()
    logger.info("Total decode + write time is: %s" % (end_val - start_val))
    return

def train(batch, model, criterion, optimizer):
    input_tensor, target, input_widths, target_widths, metadata = batch
    input_tensor = input_tensor.cuda(async=True)

    #print('train input_tensor ', input_tensor.size())
    #print('train target ', target.size(), target)
    #print('train input_widths ', input_widths.size(), input_widths)
    #print('train target_widths ', target_widths.size(), target_widths)
    #print('train metadata ', metadata)
    
    optimizer.zero_grad()
    model_output, model_output_actual_lengths = model(input_tensor, input_widths)
    
    #print('model model_output', model_output)
    #print('model model_output_actual_lengths', model_output_actual_lengths)
    #print('target widths', target_widths)
    
    #print('')
    #probs = torch.nn.functional.log_softmax(model_output.view(-1, model_output.size(2)), dim=1).view(model_output.size(0), model_output.size(1),-1)
    #log_probs = model_output.log_softmax(2).detach().requires_grad_()
    #print('log probs', log_probs)

    ''' CTC CHANGE '''
    #loss = criterion(model_output, target, model_output_actual_lengths, target_widths)
    log_probs = torch.nn.functional.log_softmax(model_output.view(-1, model_output.size(2)), dim=1).view(model_output.size(0), model_output.size(1),-1)    
    loss = criterion(log_probs, target, model_output_actual_lengths, target_widths)
    
    #print(loss.size())
    #print(loss)
    
    loss.backward()
    
    # RNN Backprop can have exploding gradients (even with LSTM), so make sure
    # we clamp the abs magnitude of individual gradient entries
    for param in model.parameters():
        if not param.grad is None:
            param.grad.data.clamp_(min=-5, max=5)


    # Okay, now we're ready to update parameters!
    optimizer.step()
    #print(loss, loss.item())
    
    ''' CTC CHANGE '''
    #return loss.data[0].item()
    return loss.item() 


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch-size", type=int, default=64, help="SGD mini-batch size")
    parser.add_argument("--max_allowed_width", type=int, default=1400, help="Max allowed width")
    parser.add_argument("--num_in_channels", type=int, default=1, help="Number of input channels for image (1 for grayscale or 3 for color)")
    parser.add_argument("--line-height", type=int, default=30, help="Input image line height")
    parser.add_argument("--rds-line-height", type=int, default=30, help="Target line height after rapid-downsample layer")
    
    parser.add_argument("--datadir", type=str, action='append', required=True, help="specify the location to data.")
    parser.add_argument("--datadirtype", type=str, default='train', help="Train type")

    parser.add_argument("--validdir", type=str, action='append', required=False, help="specify the location to data.")
    parser.add_argument("--validdirtype", type=str, default='validation', help="Train type")

    parser.add_argument("--test-datadir", type=str, default=None, help="optionally produce hyps on test set every validation pass; this is data dir")
    parser.add_argument("--test-outdir", type=str, default=None, help="optionally produce hyps on test set every validation pass; this is output dir to place hyps")
    
    parser.add_argument("--snapshot-prefix", type=str, required=True, help="Output directory and basename prefix for output model snapshots")
    parser.add_argument("--load-from-snapshot", type=str, help="Path to snapshot from which we should initialize model weights")
    parser.add_argument("--num-lstm-layers", type=int, required=True, help="Number of LSTM layers in model")
    parser.add_argument("--num-lstm-units", type=int, required=True, help="Number of LSTM hidden units in each LSTM layer (single number, or comma seperated list)")
    parser.add_argument("--lstm-input-dim", type=int, required=True, help="Input dimension for LSTM")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--nepochs", type=int, default=250, help="Maximum number of epochs to train")
    
    parser.add_argument("--snapshot-num-iterations", type=int, default=2000, help="Every N iterations snapshot model")
    parser.add_argument("--test-num-iterations", type=int, default=2000, help="Every N iterations snapshot model")
    
    parser.add_argument("--patience", type=int, default=10, help="Patience parameter for ReduceLROnPlateau.")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate for ReduceLROnPlateau")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay l2 regularization term")
    parser.add_argument("--max-val-size", type=int, default=-1, help="If validation set is large, limit it to a smaller size for faster validation runs")
    parser.add_argument("--rtl", default=False, action='store_true', help="Set if language is right-to-left")
    parser.add_argument("--synth_input", default=False, action='store_true', help="Specifies if input data is synthetic; if so we apply extra data augmentation")
    parser.add_argument("--augment", type=int, default=0, help="Add image aug library")
    parser.add_argument("--write_samples", default=False, action='store_true', help="Write sample images")
    parser.add_argument("--samples_dir", type=str, help="Samples directory")
    parser.add_argument("--cvtGray", default=False, action='store_true', help="Set if need to convert color images to grayscale") 
    
    args = parser.parse_args(argv)
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    return args

def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x, None, True)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary

def main(args):
    logger.info("Starting training\n\n")
    logger.info(torch.__version__)
    sys.stdout.flush()
    #args = get_args()
    #for arg in args:
    #    print(arg, getattr(args, arg))
 
    snapshot_path = args.snapshot_prefix + "-cur_snapshot.pth"
    best_model_path = args.snapshot_prefix + "-best_model.pth"

    line_img_transforms = []

    # Data augmentations (during training only)
    if args.synth_input:
        line_img_transforms.append( imagetransforms.DegradeDownsample(ds_factor=0.4) )


    # Make sure to do resize after degrade step above
    line_img_transforms.append(imagetransforms.Scale(new_h=args.line_height))

    # Only do for grayscale
    #if args.num_in_channels == 1:
    #    line_img_transforms.append(imagetransforms.InvertBlackWhite())

    # For right-to-left languages
    #if args.rtl:
    #    print("Right to Left")
    #    line_img_transforms.append(imagetransforms.HorizontalFlip())

    #if args.cvtGray:
    #    line_img_transforms.append(imagetransforms.ConvertGray())
 
    # add the image aug library
    if args.augment > 0:
        if args.augment == 1:
            print("Using image aug library, Level 1 ...")
            line_img_transforms.append(augment.ImageAug())
#         elif args.augment == 2:
#             print("Using image aug library, Level 2 ...")
#             line_img_transforms.append(augment_v2.ImageAug())
#         elif args.augment == 3:
#             print("Using image aug library, Level 3 ...")
#             line_img_transforms.append(augment_v3.ImageAug())
#         elif args.augment == 4:
#             print("Using image aug library, Level 4 ...")
#             line_img_transforms.append(augment_v4.ImageAug())
    else:
        print('...No augmentation')

    if args.cvtGray:
        print('...using grayscale')
        line_img_transforms.append(imagetransforms.ConvertGray())

    # Only do for grayscale
    if args.num_in_channels == 1:
        print('...invert black white')
        line_img_transforms.append(imagetransforms.InvertBlackWhite())
    
    line_img_transforms.append(imagetransforms.ToTensor())

    line_img_transforms = imagetransforms.Compose(line_img_transforms)


    # Setup cudnn benchmarks for faster code
    torch.backends.cudnn.benchmark = False

#     if len(args.datadir) == 1:
#         train_dataset = OcrDataset(args.datadir[0], args.datadirtype, line_img_transforms)
#         validation_dataset = OcrDataset(args.datadir[0], args.validdirtype, line_img_transforms)
#     else:
#         train_dataset = OcrDatasetUnion(args.datadir, args.datadirtype , line_img_transforms)
#         validation_dataset = OcrDatasetUnion(args.datadir, args.validdirtype, line_img_transforms)
    if len(args.datadir) == 1:
        train_dataset = OcrDataset(args.datadir[0], args.datadirtype, line_img_transforms, max_allowed_width=args.max_allowed_width)
        validation_dataset = OcrDataset(args.validdir[0], args.validdirtype, line_img_transforms, max_allowed_width=args.max_allowed_width)
    else:
        train_dataset = OcrDatasetUnion(args.datadir, args.datadirtype , line_img_transforms, max_allowed_width=args.max_allowed_width)
        validation_dataset = OcrDatasetUnion(args.validdir, args.validdirtype, line_img_transforms, max_allowed_width=args.max_allowed_width)


    if args.test_datadir is not None:
        if args.test_outdir is None:
            print("Error, must specify both --test-datadir and --test-outdir together")
            sys.exit(1)

        if not os.path.exists(args.test_outdir):
            os.makedirs(args.test_outdir)

        line_img_transforms_test = imagetransforms.Compose([imagetransforms.Scale(new_h=args.line_height), imagetransforms.ToTensor()])
        test_dataset = OcrDataset(args.test_datadir, "test", line_img_transforms_test)


    n_epochs = args.nepochs
    lr_alpha = args.lr
    snapshot_every_n_iterations = args.snapshot_num_iterations
    test_every_n_iterations = args.test_num_iterations
    print('...validation and snapshot iter %d' % (snapshot_every_n_iterations))
    print('...test set iter %d' % (test_every_n_iterations))

    if args.load_from_snapshot is not None:
        model = CnnOcrModel.FromSavedWeights(args.load_from_snapshot)
        print("Overriding automatically learned alphabet with pre-saved model alphabet")
        if len(args.datadir) == 1:
            train_dataset.alphabet = model.alphabet
            validation_dataset.alphabet = model.alphabet
        else:
            train_dataset.alphabet = model.alphabet
            validation_dataset.alphabet = model.alphabet
            for ds in train_dataset.datasets:
                ds.alphabet = model.alphabet
            for ds in validation_dataset.datasets:
                ds.alphabet = model.alphabet

    else:
        model = CnnOcrModel(
            num_in_channels=args.num_in_channels,
            input_line_height=args.line_height,
            rds_line_height=args.rds_line_height,
            lstm_input_dim=args.lstm_input_dim,
            num_lstm_layers=args.num_lstm_layers,
            num_lstm_hidden_units=args.num_lstm_units,
            p_lstm_dropout=0.5,
            alphabet=train_dataset.alphabet,
            multigpu=True)

    summary(model,input_size=(3, 30, 135),batch_size=32)

    # Setting dataloader after we have a chnae to (maybe!) over-ride the dataset alphabet from a pre-trained model
    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=8, sampler=GroupedSampler(train_dataset, rand=True),
                                  collate_fn=SortByWidthCollater, pin_memory=True, drop_last=True)

    if args.max_val_size > 0:
        validation_dataloader = DataLoader(validation_dataset, args.batch_size, num_workers=0,sampler=GroupedSampler(validation_dataset, max_items=args.max_val_size, fixed_rand=True),
                                           collate_fn=SortByWidthCollater, pin_memory=False, drop_last=False)
    else:
        validation_dataloader = DataLoader(validation_dataset, args.batch_size, num_workers=0,sampler=GroupedSampler(validation_dataset, rand=False),
                                           collate_fn=SortByWidthCollater, pin_memory=False, drop_last=False)



    if args.test_datadir is not None:
        test_dataloader = DataLoader(test_dataset, args.batch_size, num_workers=0,sampler=GroupedSampler(test_dataset, rand=False),
                                     collate_fn=SortByWidthCollater, pin_memory=False, drop_last=False)



    # Set training mode on all sub-modules
    model.train()

    ''' CTC CHANGE '''
    #ctc_loss = CTCLoss().cuda()
    ctc_loss = nn.CTCLoss(reduction='sum').cuda()
    #ctc_loss = nn.CTCLoss().cuda()
    
    iteration = 0
    best_val_wer = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_alpha, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, min_lr=args.min_lr)
    wer_array = []
    cer_array = []
    loss_array = []
    lr_points = []
    iteration_points = []

    epoch_size = len(train_dataloader)

    print('\n__Python VERSION:', sys.version)
    print('__PyTorch VERSION:', torch.__version__)
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Active CUDA Device: GPU', torch.cuda.current_device())
    print('__Available devices ', torch.cuda.device_count())
    print('__Current cuda device ', torch.cuda.current_device())
    print('__CUDA_VISIBLE_DEVICES %s \n' % str(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    do_test_write = True
    for epoch in range(1, n_epochs + 1):
        epoch_start = datetime.datetime.now()

        # First modify main OCR model
        for batch in train_dataloader:
            sys.stdout.flush()
            iteration += 1
            iteration_start = time.time()

            if iteration == 1 or iteration % snapshot_every_n_iterations == 0:
                print('...write samples')
                input_tensor, target, input_widths, target_widths, metadata = batch
                #print(input_widths, target_widths, metadata)
                image_list = input_tensor.numpy()   
                for img_ctr in range(0, len(image_list)):
                    if img_ctr < 10:
                        img = np.uint8(image_list[img_ctr,:].squeeze().transpose((1,2,0)) * 255)
                        cv2.imwrite(args.samples_dir + '/' + str(iteration) + '_' + str(img_ctr) + ".png", img)
                    else:
                        break
            
            loss = train(batch, model, ctc_loss, optimizer)

            #elapsed_time = datetime.datetime.now() - iteration_start
            duration = time.time() - iteration_start
            loss = loss / args.batch_size

            #loss_array.append(loss)

            if iteration % 10 == 0:
                input_tensor, target, input_widths, target_widths, metadata = batch
                #print(input_widths, target_widths, target)
                examples_per_sec = args.batch_size / duration
                sec_per_batch = float(duration)
                logger.info("Iteration: %d (%d/%d in epoch %d), Batch Size: %d, Max Width: %d, Loss: %f, LR: %f, ex/sec: %.1f, sec/batch: %.2f" % (
                    iteration, iteration % epoch_size, epoch_size, epoch, args.batch_size, input_widths[0], loss, lr_alpha, examples_per_sec, sec_per_batch))

            # Do something with loss, running average, plot to some backend server, etc

            #if iteration == 1 or iteration % snapshot_every_n_iterations == 0:
            if iteration % snapshot_every_n_iterations == 0:
                logger.info("Testing on validation set")
                val_loss, val_cer, val_wer = test_on_val(validation_dataloader, model, ctc_loss, args.samples_dir)

                if val_cer < 0.5:
                    do_test_write = True
                
                if iteration % test_every_n_iterations == 0:
                    print('...Time for test eval %d' % (iteration))
                                     
                if args.test_datadir is not None and (iteration % test_every_n_iterations == 0) and do_test_write:
                    print('......start test eval')
                    out_hyp_outdomain_file = os.path.join(args.test_outdir, "hyp-%07d.outdomain.utf8" % iteration)
                    out_hyp_indomain_file = os.path.join(args.test_outdir, "hyp-%07d.indomain.utf8" % iteration)
                    out_meta_file = os.path.join(args.test_outdir, "hyp-%07d.meta" % iteration)
                    test_on_val_writeout(test_dataloader, model, out_hyp_outdomain_file)
                    test_on_val_writeout(validation_dataloader, model, out_hyp_indomain_file)
                    with open(out_meta_file, 'w') as fh_out:
                        fh_out.write("%d,%f,%f,%f\n" % (iteration, val_cer, val_wer, val_loss))
                
                # Reduce learning rate on plateau
                early_exit = False
                lowered_lr = False
                if scheduler.step(val_wer):
                    lowered_lr = True
                    lr_points.append(iteration / snapshot_every_n_iterations)
                    if scheduler.finished:
                        early_exit = True

                    # for bookeeping only
                    lr_alpha = max(lr_alpha * scheduler.factor, scheduler.min_lr)

                logger.info("Val Loss: %f\tNo LM Val CER: %f\tNo LM Val WER: %f" % (val_loss, val_cer, val_wer))

                torch.save({'iteration': iteration,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_hyper_params': model.get_hyper_params(),
                            'rtl': args.rtl,
                            'cur_lr': lr_alpha,
                            'line_height': args.line_height
                        },
                           snapshot_path)

                # plotting lr_change on wer, cer and loss.
                wer_array.append(val_wer)
                cer_array.append(val_cer)
                iteration_points.append(iteration / snapshot_every_n_iterations)

                if val_wer < best_val_wer:
                    logger.info("Best model so far, copying snapshot to best model file")
                    best_val_wer = val_wer
                    shutil.copyfile(snapshot_path, best_model_path)

                logger.info("Running WER: %s" % str(wer_array))
                logger.info("Done with validation, moving on.")

                if early_exit:
                    logger.info("Early exit")
                    sys.exit(0)

                if lowered_lr:
                    logger.info("Switching to best model parameters before continuing with lower LR")
                    weights = torch.load(best_model_path)
                    model.load_state_dict(weights['state_dict'])


        elapsed_time = datetime.datetime.now() - epoch_start
        logger.info("\n------------------")
        logger.info("Done with epoch, elapsed time = %s" % pretty_print_timespan(elapsed_time))
        logger.info("------------------\n")


    #writer.close()
    logger.info("Done.")


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
