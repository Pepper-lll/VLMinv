# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Code written by Pavlo Molchanov and Hongxu Yin
# --------------------------------------------------------

import torch
import os
from torch import distributed, nn
import random
import numpy as np
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import renew_vae_resnet_paths, assign_to_checkpoint,\
    conv_attn_to_linear, renew_vae_attention_paths
import spacy
from PIL import Image
from copy import deepcopy
from torch.nn import Module

class AveragedPara(Module):
    def __init__(self, paras, device=None, avg_fn=None):
        super(AveragedPara, self).__init__()
        self.paras = deepcopy(paras)
        if device is not None:
            self.paras = self.paras.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged.item() + 1)
        self.avg_fn = avg_fn
    
    def update_parameters(self, paras):
        for p_swa, p_now in zip(self.paras, paras):
            device = p_swa.device
            p_now_ = p_now.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_now_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_now_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1

def blip_eva(images, texts, result_dict, model, vis_process):
    # model = model.cuda()
    nlp = spacy.load("en_core_web_sm")
    batch_avg_score = 0.0
    for id in range(images.shape[0]):
        noun_list = list(nlp(texts[id]).noun_chunks)
        question_list = []
        for noun in noun_list:
            question_list.append(str(noun)+'?')
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        image = vis_process["eval"](pil_image).unsqueeze(0)
        samples = {"image": image, "text_input": " "}
        blip_score = 1.0
        prob_list = []
        for q in question_list:
            prob = model.predict_answers(samples, question_list=q, inference_method="vqa_prob")
            prob_list.append(prob[0])
            blip_score *=prob[0]
        
        batch_avg_score += blip_score
        result_dict[texts[id]] = [blip_score, question_list, prob_list]
    batch_avg_score /= images.shape[0]
    return batch_avg_score

def load_model_pytorch(model, load_model, gpu_n=0):
    print("=> loading checkpoint '{}'".format(load_model))

    checkpoint = torch.load(load_model, map_location = lambda storage, loc: storage.cuda(gpu_n))

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    if 1:
        if 'module.' in list(model.state_dict().keys())[0]:
            if 'module.' not in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

        if 'module.' not in list(model.state_dict().keys())[0]:
            if 'module.' in list(load_from.keys())[0]:
                from collections import OrderedDict

                load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    if 1:
        if list(load_from.items())[0][0][:2] == "1." and list(model.state_dict().items())[0][0][:2] != "1.":
            load_from = OrderedDict([(k[2:], v) for k, v in load_from.items()])

        load_from = OrderedDict([(k, v) for k, v in load_from.items() if "gate" not in k])

    model.load_state_dict(load_from, strict=True)

    epoch_from = -1
    if 'epoch' in checkpoint.keys():
        epoch_from = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(load_model, epoch_from))


def create_folder(directory):
    # from https://stackoverflow.com/a/273227
    if not os.path.exists(directory):
        os.makedirs(directory)


random.seed(0)

def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def scale_linear_policy(epoch, base_scale, final_scale, start_epoch, end_epoch):
    step = (final_scale - base_scale) / (end_epoch - start_epoch)
    if epoch in range(start_epoch, end_epoch):
        scale = base_scale + step
    elif epoch < start_epoch:
        scale = base_scale
    else:
        scale = final_scale
    return scale

def scale_cosine_policy(epoch, base_scale, final_scale, start_epoch, end_epoch):
    if epoch in range(start_epoch, end_epoch):
        e = epoch - start_epoch
        es = end_epoch - start_epoch
        scale = 0.5 * (1 + np.cos(np.pi * e / es)) * (base_scale - final_scale) + final_scale
    elif epoch < start_epoch:
        scale = base_scale
    else:
        scale = final_scale
    return scale

def sd_weight_policy(t, min_t, max_t):
    ts = max_t - min_t
    weight = 0.25 * (np.cos(2*np.pi * (t-min_t) / ts)) + 0.75
    return weight

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs, min_lr=0.0):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr) + min_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    # for key in keys:
    #     if key.startswith(vae_key):
    #         vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)
    for key in keys:
        vae_state_dict[key] = checkpoint.get(key)

    new_checkpoint = {}
    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint