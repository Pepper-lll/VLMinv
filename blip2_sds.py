
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.cuda.amp as amp
import random
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from torch.nn import functional as F
import torch.distributed as dist
import sys
import datetime
from utils import blip_eva, lr_cosine_policy, scale_cosine_policy, AveragedPara, scale_linear_policy, clip, denormalize, create_folder
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import create_vae_diffusers_config
from omegaconf import OmegaConf
import pickle
import spacy
import json
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
from lavis.common.registry import registry
from lavis.models import load_preprocess

from torchvision import transforms
import torchvision.models as models
from torchvision.transforms.functional import InterpolationMode

from typing import List, Optional
from matplotlib import pyplot as plt
import yaml

from diffusers import AutoencoderKL, VQModel, StableDiffusionPipeline
from vqgan_utils import *
from sds_utils import SDSLoss, get_text_embeddings, decode

from tqdm import tqdm



def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

blip2_normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

class BLIP2Inv_SDS(object):
    def __init__(self, gen_idx=0, bs=84, save_every=5, save=False, aug_topk=10,
                 noise_scale=0.002, z_noise=0.0, cutn=10, augs=['Af', 'Pe', 'Ji', 'Er'],
                 seed=0, blip_times=1,
                 alpha_exp=0, sigma_exp=0,
                 lm_weight=1.0, itm_weight=1.0,
                 blip_weight=1.0, sd_weight=2000.0,
                 sd_scale=False, blip_scale=False,
                 blip_mul_final=20.0, sd_mul_final=800.0,
                 blip_scale_iter=[0, 400], sd_scale_iter=[100, 400],
                 start_images=None, 
                 lr_sch=100, lrsch=False, sd_t_range=[50, 950], min_lr=0.0,
                 ema_decay=0.95, blip_range=[0, 100],
                 guidance_scale=10.0,
                 itg_weight=1., optimizer='Adam', wd=5e-4, betas=[0.9, 0.999], momentum=0.999,
                 ema_start = 20, ema_restart = 5,
                 img_size=224,
                 use_fp16=True, net=None, text_input=None,
                 text_idx=None, recon_blip_score_dic={},
                 path="./gen_images/",
                 epochs=10000,
                 coefficients=dict(),):


        # for reproducibility
        # torch.manual_seed(torch.cuda.current_device())
        torch.manual_seed(seed)

        self.gen_idx=gen_idx
        self.blip_times=blip_times
        self.net = net
        self.itg_weight = itg_weight
        self.epochs=epochs
        self.optimizer = optimizer
        self.wd=wd
        self.betas=betas
        self.momentum=momentum
        self.ema_start=ema_start
        self.ema_restart=ema_restart
        self.start_images=start_images
        if start_images is not None:
            self.start_images=start_images.cuda()
        self.noise_scale=noise_scale
        self.z_noise=z_noise
        self.cutn=cutn
        self.aug_topk=aug_topk
        self.augs=augs
        self.sd_t_range=sd_t_range
        self.guidance_scale=guidance_scale
        self.ema_decay=ema_decay
        self.blip_range=blip_range
        self.blip_weight=blip_weight
        self.sd_weight=sd_weight
        self.blip_scale=blip_scale
        self.sd_scale=sd_scale
        self.blip_mul_final=blip_mul_final
        self.sd_mul_final=sd_mul_final
        self.blip_scale_iter=blip_scale_iter
        self.sd_scale_iter=sd_scale_iter

        self.image_resolution = img_size
        self.alpha_exp=alpha_exp
        self.sigma_exp=sigma_exp
        self.itm_weight=itm_weight
        self.lm_weight=lm_weight

        self.bs = bs  # batch size
        self.use_fp16 = use_fp16
        self.save = save
        self.save_every = save_every
        # self.criterion = criterion
        # self.main_loss_multiplier = main_loss_multiplier
        self.lr_sch = lr_sch
        self.lrsch = lrsch
        self.min_lr = min_lr
        self.recon_blip_score_dic = recon_blip_score_dic
        self.text_input = text_input
        self.text_idx = text_idx
        
        self.var_scale_l1 = coefficients["tv_l1"]
        self.var_scale_l2 = coefficients["tv_l2"]
        self.l2_scale = coefficients["l2"]
        self.lr = coefficients["lr"]
        self.main_loss_multiplier = coefficients["main_loss_multiplier"]

        self.num_generations = 0

        ## Create folders for images and logs
        prefix = path
        self.prefix = prefix

        local_rank = torch.cuda.current_device()
        if self.save >0 and local_rank==0:
            create_folder(prefix + '/img/')
            

    def get_images(self):
        torch.autograd.set_detect_anomaly(True)
        print("get_images call")
        print(self.text_idx, self.text_input)

        net = self.net
        text_input = self.text_input
        text_idx = self.text_idx
        use_fp16 = self.use_fp16
        save_every = self.save_every

        local_rank = torch.cuda.current_device()
        img_original = self.image_resolution
    
        ## Evaluation model loading
        model_name="blip_vqa"
        model_cls = registry.get_model_class(model_name)
        print(model_cls)
        model_config_path = "blip2/blip_vqav2.yaml"
        model_cfg = OmegaConf.load(model_config_path).model
        model = model_cls.from_config(model_cfg)
        model.eval()
        blip_vqa_preprocess_cfg = OmegaConf.load(model_config_path).preprocess
        blipvqa_vis_process, _ = load_preprocess(blip_vqa_preprocess_cfg)
        
        if self.use_fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.device('cuda')
        model_id  =  "runwayml/stable-diffusion-v1-5"
        # model_id = "/home/ma-user/work/lxt/sd-v2-1"
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder
        vae = pipeline.vae
        # vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae', torch_dtype=torch.float32)
        # breakpoint()
        sds_loss = SDSLoss(device=device, pipe=pipeline, dtype=dtype, t_min=self.sd_t_range[0], t_max=self.sd_t_range[1], alpha_exp=self.alpha_exp, sigma_exp=self.sigma_exp)
        del pipeline
        
        iteration = 0
        losses = {
            'itm_loss':[], 'lm_loss':[], 'im_sim':[]
        }
        
        g=torch.Generator()
        if self.gen_idx > 0:
            seed_ = np.random.randint(5000, size=1)
            print('Seeds:', seed_)
            g.manual_seed(int(seed_))
            if use_fp16:
                z = torch.randn(size=(1, 4, int(img_original/8), int(img_original/8)), generator=g).to(device, dtype=torch.float16)
            else:
                z = torch.randn(size=(1, 4, int(img_original/8), int(img_original/8)), generator=g).to(device)
        else:
            if use_fp16:
                z = torch.randn(size=(1, 4, int(img_original/8), int(img_original/8)), dtype=torch.float16).to(device)
            else:
                z = torch.randn(size=(1, 4, int(img_original/8), int(img_original/8))).to(device)

        with torch.no_grad():
            embedding_null = get_text_embeddings(tokenizer, text_encoder, "")
            embedding_text_target = get_text_embeddings(tokenizer, text_encoder, "A photo exactly contains "+text_input[0])
            embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)
        z.requires_grad = True
        # image_target.requires_grad = True
        # breakpoint()
        print('z shape:', z.shape)
        
        if self.ema_decay > 0.0:
            # z_ema = torch.utils.ExponentialMovingAverage(z, device=z.device, decay=self.ema_decay)
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                     self.ema_decay * averaged_model_parameter + (1.0-self.ema_decay) * model_parameter
            z_ema = AveragedPara(z, avg_fn=ema_avg)
        
        iterations_per_layer = self.epochs
        if self.optimizer == 'SGD':
            optimizer = optim.SGD([z], lr=self.lr)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam([z], lr=self.lr, betas=self.betas, weight_decay=self.wd)
        else:
            print('Unsupported optimizer !!!')
        print('optimizer:', optimizer)
        lr_scheduler = lr_cosine_policy(self.lr, self.lr_sch, iterations_per_layer, self.min_lr)
        make_cutouts = MakeCutouts(224, self.cutn, self.augs)  
        # clamp_with_grad = ClampWithGrad.apply/
        print(' #### Start training ####')
        start_time = datetime.datetime.now()
        
        blip_weight = self.blip_weight
        sd_weight = self.sd_weight
        
        
        
        for iteration_loc in tqdm(range(iterations_per_layer)):
            if self.blip_scale:
                # blip_weight = blip_weight - 0.5
                # print('!!!!blip_weight:', blip_weight)
                blip_weight = scale_cosine_policy(iteration_loc, blip_weight, self.blip_mul_final, self.blip_scale_iter[0], self.blip_scale_iter[1])
            if self.sd_scale:
                sd_weight = scale_cosine_policy(iteration_loc, self.sd_weight, self.sd_mul_final, self.sd_scale_iter[0], self.sd_scale_iter[1])
            iteration += 1
            # learning rate scheduling
            if self.lrsch:
                lr_scheduler(optimizer, iteration_loc, iteration_loc)
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                # print('lr:', lr)
            sd_loss, log_loss, __, __ = sds_loss.get_sds_loss(z=z, text_embeddings=embedding_target, guidance_scale=self.guidance_scale)
            z_scale_factor = 0.18215
            image_target = vae.decode((1 / z_scale_factor) * z, return_dict=False)[0]
            image_target = (image_target / 2 + 0.5).clamp(0, 1)
            if self.ema_decay > 0.0:
                ema_img_target = vae.decode((1 / z_scale_factor) * z_ema.paras, return_dict=False)[0]
                ema_img_target = (ema_img_target / 2 + 0.5).clamp(0, 1)
            # breakpoint()
            loss_itm = torch.tensor([0.0]).to(z.device)
            loss_lm = torch.tensor([0.0]).to(z.device)
            loss_sim = torch.tensor([0.0]).to(z.device)
            blip_resize = transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                )
            # if iteration_loc >=self.blip_range[1]:
            #     sd_weight=4000
            if iteration_loc in range(self.blip_range[0], self.blip_range[1]):
                cutouts4all = []
                if self.cutn > 1:
                    for input in image_target:
                        cutouts4one = make_cutouts(input.unsqueeze(0))
                        cutouts4all.append(cutouts4one)
                    imgs = torch.cat(cutouts4all, dim=0)
                    cur_inputs = blip2_normalize(blip_resize(imgs))
                    with torch.no_grad():
                        aug_sim_loss, aug_itm_loss, aug_lm_loss = self.get_blip_loss(cur_inputs, net, text_input, blip_resize, loss_reduct='sep', losses=None)
                        aug_losses = aug_sim_loss+aug_itm_loss+aug_lm_loss
                    imgs = imgs[aug_losses.topk(self.aug_topk)[1]]
                else:
                    imgs = image_target
                inputs_jit = blip2_normalize(imgs + self.noise_scale * torch.randn_like(imgs))
                
                loss_sim, loss_itm, loss_lm = self.get_blip_loss(inputs_jit, net, text_input, blip_resize, losses)
                

            sd_loss = sd_weight * sd_loss
            # print("sd_weight:", sd_weight)
            # if self.blip_scale and iteration_loc == 200 and self.blip_range[0] <= 200:
            #     self.blip_weight *= 10.0
            if iteration_loc in range(self.blip_range[0], self.blip_range[1]):
                blip_loss = blip_weight * (loss_itm + loss_lm)
                loss = sd_loss + blip_loss
            else:
                loss = sd_loss
            
            optimizer.zero_grad()
            net.zero_grad()
            
            grad1 = torch.autograd.grad((sd_loss), z)[0]
            if iteration_loc in range(self.blip_range[0], self.blip_range[1]):
                grad2= torch.autograd.grad(blip_loss, z)[0]
                grad1_norm = torch.norm(grad1.reshape(z.shape[0], -1), dim=1).mean()
                grad2_norm = torch.norm(grad2.reshape(z.shape[0], -1), dim=1).mean()

                grad_scale = grad1_norm / grad2_norm
                z.grad = blip_weight * grad_scale * grad2 + self.z_noise * torch.randn_like(z)
                optimizer.step()
                grad1 = grad1
                z.data = z.data - self.lr * grad1 + self.z_noise * torch.randn_like(z)
            else:
                # z.grad = grad1
                grad1 = grad1
                z.data = z.data - self.lr * grad1 + 0.01 * torch.randn_like(z)

            
            if self.ema_decay > 0.0:
                z_ema.update_parameters(z)
                if (iteration_loc+1-self.ema_start) >=0 and (iteration_loc+1-self.ema_start) % self.ema_restart == 0 and (iteration_loc+1)<=self.blip_range[1]:
                    z.data = z_ema.paras
                    
            image_target = vae.decode((1 / z_scale_factor) * z, return_dict=False)[0]
            image_target = (image_target / 2 + 0.5).clamp(0, 1)    
            z_scale_factor = z.std().item()
            if iteration_loc in range(self.blip_range[0], self.blip_range[1]):
                loss = sd_loss + blip_loss
            else:
                loss = sd_loss
            
            if local_rank==0:
                if iteration_loc % 10 == 0 or iteration_loc==iterations_per_layer-1 or (iteration_loc+1)==self.blip_range[1]:
                    now_time = datetime.datetime.now()
                    # batch_blip_score = 0.0
                    batch_blip_score = blip_eva(image_target.detach(), text_input, self.recon_blip_score_dic, model, blipvqa_vis_process)
                    print("------------iteration {}----------".format(iteration_loc))
                    print('recon batch average blip vqa score:', batch_blip_score)
                    print("total loss", loss.item())
                    print("image text matching loss", loss_itm.item())
                    print("image text captioning loss", loss_lm.item())
                    print("sd loss {}".format(sd_loss.item()))
                    print("sd weight {}".format(sd_weight))
                    print('Now time:', now_time)
                    print('*** Time cost: ***', now_time - start_time)
                
            if (iteration_loc==iterations_per_layer-1) or (iteration_loc+1)==self.blip_range[1] and self.save:
                if local_rank==0:
                    vutils.save_image(image_target,
                                    '{}/batch_img/{}_{:04d}_{:04f}_{}_{:02d}.png'.format(self.prefix, text_idx, iteration_loc, batch_blip_score, text_input[0][:50], self.gen_idx),
                                    normalize=True, scale_each=True, nrow=int(10))
                    

        print('#### End training ####')
        end_time = datetime.datetime.now()
        if self.save:
            with open(self.prefix+"/out.txt", 'a') as out_txt:
                out_txt.write('*** Total time cost: {} ***\n\n'.format(end_time - start_time))
        print('*** Total time cost: ***', end_time - start_time)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        return batch_blip_score


    def new_method(self, outputs):
        outputs = self.network_output_function(outputs)
        return outputs

    def save_images(self, images, texts, iteration):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            text = texts[id]
            place_to_store = '{}/img/{:05d}_{}_gpu_{}.jpg'.format(self.prefix, iteration, text, local_rank)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

    def generate_batch(self,):
        net = self.net

        recon_blip_score = self.get_images()

        net.eval()

        self.num_generations += 1
        return recon_blip_score
    
    def get_blip_loss(self, inputs_jit, net, text_input, blip_resize, losses, loss_reduct='mean'):
        with torch.cuda.amp.autocast(enabled=(inputs_jit.device != torch.device("cpu"))):
            img_embedding = net.ln_vision(net.visual_encoder(blip_resize(inputs_jit)))
        
        # breakpoint()
        image_atts = torch.ones(img_embedding.size()[:-1], dtype=torch.long).to(
            inputs_jit.device
        )
        repeat_text_input = []
        for i in range(len(text_input)):
            repeat_text_input.extend([text_input[i]]*inputs_jit.shape[0])
        
        text = net.tokenizer(
            repeat_text_input,
            padding="max_length",
            truncation=True,
            max_length=net.max_txt_len,
            return_tensors="pt",
        ).to(inputs_jit.device)
        text_output = net.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        # print('text_output:', text_output.last_hidden_state.shape)
        text_feat = F.normalize(
            net.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        query_tokens = net.query_tokens.expand(img_embedding.shape[0], -1, -1)
        query_output = net.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=img_embedding,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        image_feats = F.normalize(
                    net.vision_proj(query_output.last_hidden_state), dim=-1
                )
        # print('image_feats:', image_feats.shape)
        image_feats_all = concat_all_gather(image_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        ###============== Image-text Similarity ===================###
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()
        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        if image_feats.shape[0] > 1:
            if loss_reduct == 'sep': 
                sim_ti = torch.diag(sim_t2i)
            else:
                sim_ti = torch.diag(sim_t2i).mean()
        else:
            sim_ti = sim_t2i
        
        # loss_sim = (1-sim_ti + 1-sim_it) / 2
        loss_sim = 1- sim_ti
        if losses:
            losses["im_sim"].append(loss_sim.item())
        # breakpoint()
        
        ###============== Image-text Matching ===================###
        if self.itm_weight > 0.0:
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                inputs_jit.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            output_itm = net.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=img_embedding,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
            itm_logit = net.itm_head(itm_embeddings)
            itm_logit = itm_logit.mean(dim=1)
            itm_prob = torch.nn.functional.softmax(itm_logit, dim=1)
            if loss_reduct == 'sep':
                loss_itm = -torch.log(itm_prob[:, 1] + 1e-8)
            else:
                loss_itm = -torch.log(itm_prob[:, 1] + 1e-8).mean()
                
        # breakpoint()

        ##================= Image Captioning ========================##
        if self.lm_weight > 0.0:
            decoder_input_ids = text.input_ids.clone()
            decoder_input_ids[:, 0] = net.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == net.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                inputs_jit.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            if loss_reduct == 'sep': ## TODO
                lm_output = net.Qformer(
                    decoder_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=query_output.past_key_values,
                    return_dict=True,
                    labels=labels,
                    reduction="none",
                )
                loss_lm = lm_output.loss / (labels.shape[-1]-1)
            else:
                lm_output = net.Qformer(
                    decoder_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=query_output.past_key_values,
                    return_dict=True,
                    labels=labels,
                )
                # breakpoint()
                loss_lm = lm_output.loss
            
        return loss_sim, loss_itm, loss_lm
