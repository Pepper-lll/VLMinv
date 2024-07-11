from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import sys
# sys.path.append('/home/ma-user/work/lxt/packages')
import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import torch.cuda.amp as amp
import os

os.environ["USE_TORCH"]= "True"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torchvision.models as models
from utils import load_model_pytorch, distributed_is_initialized

from PIL import Image
from lavis.common.registry import registry
from lavis.models import load_model_and_preprocess, load_preprocess
from lavis.processors import load_processor
from omegaconf import OmegaConf
import datetime
from utils import create_folder
import json

from blip2_sds import BLIP2Inv_SDS

    
def vae_process(raw_image, img_size):
    image = raw_image.resize((img_size, img_size))
    image = np.array(image).astype(np.float32) / 127.5 - 1.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def run(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print('Loading pretrained blip2 image text matching model...')
    
    ############## Load model locally ##############
    name = "blip2_image_text_matching"
    model_cls = registry.get_model_class(name)
    cfg = OmegaConf.load('blip2/blip2_pretrain.yaml')
    cfg_model = cfg.model
    vit_model = cfg_model.get("vit_model", "eva_clip_g")
    img_size = cfg_model.get("image_size")
    num_query_token = cfg_model.get("num_query_token")
    cross_attention_freq = cfg_model.get("cross_attention_freq", 2)
    drop_path_rate = cfg_model.get("drop_path_rate", 0)
    use_grad_checkpoint = cfg_model.get("use_grad_checkpoint", False)
    vit_precision = cfg_model.get("vit_precision", "fp16")
    freeze_vit = cfg_model.get("freeze_vit", True)
    max_txt_len = cfg_model.get("max_txt_len", 32)
    net = model_cls(
        vit_model=vit_model,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision,
        freeze_vit=freeze_vit,
        num_query_token=num_query_token,
        cross_attention_freq=cross_attention_freq,
        max_txt_len=max_txt_len,
    )
    #load pretrained Q-former paras.
    #对于visual encoder会显示'unmatched keys'，因为load的只有Qformer，但是ViT model初始化已经是pretrain好的，所以不用管
    net.load_from_pretrained("blip2/blip2_pretrained.pth")
    process_cfg = OmegaConf.load('blip2/blip2_pretrain.yaml').preprocess
    vis_processors, text_processors = load_preprocess(process_cfg)
    print('Finish loading BLIP2 model and processors.')
    

    lines = ["“Two hot dogs sit on a white paper plate near a soda cup which is sitting on a green picnic table while a bike and a silver car are parked nearby",
             "A red backpack and a blue book"]
    text_input = [text_processors["eval"](line) for line in lines]
    text_idxs = [i for i in range(len(text_input))]
    
    net = net.to(device)
    net.eval()
    
    time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S').replace('  ','-')
    print('Start runing time:', time)
    
    exp_name = args.exp_name + '/'
    
    
    if args.save:
        create_folder(exp_name)
        args_dict = vars(args)
        json.dump(args_dict, open(exp_name+'/args.json', 'w'))
        
        argsDict = args.__dict__
        with open(exp_name+"/out.txt", 'a') as out_txt:
            for eachArg, value in argsDict.items():
                out_txt.writelines(eachArg + ' : ' + str(value) + '\n')
            out_txt.write('\n')
    # args.detach_student = False

    # args.resolution = 224
    bs = args.bs
    coefficients = dict()
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["l2"] = args.l2
    coefficients["lr"] = args.lr
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    # coefficients["adi_scale"] = args.adi_scale

    
    for i in range(int(len(text_input) / args.bs)):
        best_tmp = 0.0
        if (i+1)*args.bs > len(text_input):
            cur_texts = text_input[i*args.bs : len(text_input)]
        else:
            cur_texts = text_input[i*args.bs : (i+1)*args.bs]
        cur_idx = text_idxs[i*args.bs: (i+1)*args.bs]
        print("Current prompt:", cur_idx, cur_texts)
        if args.save:
            with open(exp_name+"/out.txt", 'a') as out_txt:
                out_txt.write('Current prompt {} {} \n'.format(cur_idx, cur_texts))
        # text_idx=str(i*args.bs)+'-'+str((i+1)*args.bs)={}
        recon_blip_score_dic={}
        for num in range(args.gen_num):
            Engine = BLIP2Inv_SDS(net=net, gen_idx=num, blip_times=args.blip_times, text_input=cur_texts, text_idx=str(cur_idx), cutn=args.cutn, 
                                                    seed=args.seed, augs=args.augs, aug_topk=args.aug_topk,
                                                    blip_weight=args.blip_weight, sd_weight=args.sd_weight, blip_scale=args.blip_scale, sd_scale=args.sd_scale,
                                                    blip_mul_final=args.blip_mul_final, sd_mul_final=args.sd_mul_final,
                                                    blip_scale_iter=args.blip_scale_iter, sd_scale_iter=args.sd_scale_iter,
                                                    ema_decay=args.ema_decay, blip_range=args.blip_range,
                                                    sd_t_range=args.sd_t_range,  
                                                    guidance_scale=args.guidance_scale,
                                                    alpha_exp=args.alpha_exp, sigma_exp=args.sigma_exp,
                                                    lm_weight=args.lm_weight, itm_weight=args.itm_weight,
                                                    noise_scale=args.noise_scale, z_noise=args.z_noise,
                                                    ema_start=args.ema_start, ema_restart=args.ema_restart,
                                                    lr_sch=args.lr_sch, start_images = None, min_lr=args.min_lr,
                                                    optimizer=args.optimizer, wd=args.wd, betas=args.betas, momentum=args.momentum,
                                                    recon_blip_score_dic=recon_blip_score_dic,
                                                    lrsch=args.lrsch, epochs=args.epochs,
                                                    itg_weight=args.itg_weight,
                                                    path=exp_name, img_size=args.img_size,
                                                    bs = bs, save_every=args.save_every, save=args.save,
                                                    use_fp16 = args.fp16,
                                                    coefficients = coefficients)
            recon_blip_score = Engine.generate_batch()
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--extra_info', default='', type=str)
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--betas', type=float, nargs='*', default=[0.9, 0.999], help="")
    parser.add_argument('--ema_start', default=40, type=int)
    parser.add_argument('--ema_restart', default=60, type=int)
    parser.add_argument('--epochs', default=160, type=int, help='')
    
    parser.add_argument('--lrsch', default=True)
    parser.add_argument('--lr_sch', type=int, default=10, help='')
    parser.add_argument('--save', action='store_true', help='save images')
    parser.add_argument('--save_every', default=500, type=int, help='save images every # epochs')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='output/', help='where to store experimental data')
    
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate for optimization')
    parser.add_argument('--min_lr', type=float, default=0.5, help='')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--itg_weight', type=float, default=1., help='weight for image-grounded text generation loss')
    
    parser.add_argument('--noise_scale', type=float, default=0.2)
    parser.add_argument('--z_noise', type=float, default=0.2)
    
    parser.add_argument('--guidance_scale', type=float, default=30.0, help="diffusion model classifier-free guidance scale")
    parser.add_argument('--sd_t_range', type=int, nargs='*', default=[50, 950], help="stable diffusion time steps range")
    parser.add_argument('--ema_decay', default=0.95, type=float)
    parser.add_argument("--blip_range",  type=int, nargs='*', default=[0, 150],)
    parser.add_argument('--blip_weight', type=float, default=2.0, help="")
    parser.add_argument('--sd_weight', type=float, default=800.0, help="")
    parser.add_argument('--blip_scale', default=False)
    parser.add_argument('--sd_scale', default=True)
    parser.add_argument("--blip_scale_iter",  type=int, nargs='*', default=[0, 100],)
    parser.add_argument("--sd_scale_iter",  type=int, nargs='*', default=[0, 160],)
    parser.add_argument('--blip_mul_final', type=float, default=30.0, help="")
    parser.add_argument('--sd_mul_final', type=float, default=400.0, help="")
    parser.add_argument("--alpha_exp",  type=int, default=0,)
    parser.add_argument('--sigma_exp', type=int, default=0, help="")
    parser.add_argument('--lm_weight', type=float, default=1.0, help='')
    parser.add_argument('--itm_weight', type=float, default=1.0, help='')
    
    parser.add_argument("--augs", type=str, nargs='+', help="Enabled augments (latest vut method only)", default=['Af','Pe','Ji','Er'])
    parser.add_argument("--cutn",  type=int, default=10)
    parser.add_argument("--aug_topk", type=int, help="", default=10)
    
    parser.add_argument("--gen_num",  type=int, default=2)
    parser.add_argument("--blip_times",  type=int, default=1)
    args = parser.parse_args()
    # print('print redirected')
    # sys.stdout = open(args.exp_name + '/output.txt','wt')
    print(args)

    torch.backends.cudnn.benchmark = True

    run_cfg = OmegaConf.create()
    run_cfg.dist_url = args.dist_url
    init_distributed_mode(run_cfg)

    run(args)


if __name__ == '__main__':
    main()