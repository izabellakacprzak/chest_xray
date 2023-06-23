import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

from dscmchest.pgm_chest.train_pgm import preprocess
from dscmchest.pgm_chest.flow_pgm import FlowPGM
from dscmchest.vae_chest import HVAE
from dscmchest.pgm_chest.dscm import DSCM

class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

def load_chest_models():
    # Load predictors
    args = Hparams()
    args.predictor_path = '/homes/iek19/Documents/FYP/chest_xray/checkpoints/a_r_s_f/mimic_classifier_resnet18_l2_slurm/checkpoint.pt'
    predictor_checkpoint = torch.load(args.predictor_path)
    args.update(predictor_checkpoint['hparams'])
    predictor = FlowPGM(args)
    
    # Load PGM
    args.pgm_path = '/homes/iek19/Documents/FYP/chest_xray/checkpoints/a_r_s_f/sup_pgm_mimic/checkpoint.pt'
    # print(f'\nLoading PGM checkpoint: {args.pgm_path}')
    pgm_checkpoint = torch.load(args.pgm_path)
    pgm_args = Hparams()
    pgm_args.update(pgm_checkpoint['hparams'])
    pgm = FlowPGM(pgm_args)
    # pgm.load_state_dict(pgm_checkpoint['ema_model_state_dict'])

    args.vae_path = '/homes/iek19/Documents/FYP/chest_xray/checkpoints/a_r_s_f/mimic_beta9_gelu_dgauss_1_lr3/checkpoint.pt'

    # print(f'\nLoading VAE checkpoint: {args.vae_path}')
    vae_checkpoint = torch.load(args.vae_path)
    vae_args = Hparams()
    vae_args.update(vae_checkpoint['hparams'])
    vae = HVAE(vae_args)
    # vae.load_state_dict(vae_checkpoint['ema_model_state_dict'])

    args = Hparams()
    dscm_dir = "mimic_dscm_lr_1e5_lagrange_lr_1_damping_10"
    # dscm_dir = "mimic_more_dscm_lr_1e5_lagrange_lr_1_damping_10"

    which_checkpoint="6500_checkpoint"
    args.load_path = f'/homes/iek19/Documents/FYP/chest_xray/checkpoints/a_r_s_f/{dscm_dir}/{which_checkpoint}.pt'
    # print(args.load_path)
    dscm_checkpoint = torch.load(args.load_path)
    args.update(dscm_checkpoint['hparams'])
    model = DSCM(args, pgm, predictor, vae)
    args.cf_particles =1
    model.load_state_dict(dscm_checkpoint['model_state_dict'])
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(args.device)
    # Set model require_grad to False
    for p in model.parameters():
        p.requires_grad = False
    del vae, pgm, predictor
    return model, args, pgm_args

model, _, _ = load_chest_models()

def norm(batch):
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = (batch['x'].float() - 127.5) / 127.5  # [-1,1]
        elif k in ['age']:
            batch[k] = batch[k].float().unsqueeze(-1)
            batch[k] = batch[k] / 100.
            batch[k] = batch[k] * 2 - 1  # [-1,1]
        elif k in ['race']:
            batch[k] = F.one_hot(batch[k], num_classes=3).squeeze().float()
        elif k in ['finding']:
            batch[k] = batch[k].unsqueeze(-1).float()
        else:
            batch[k] = batch[k].float().unsqueeze(-1)
    return batch

def postprocess(x):
    return ((x + 1.0) * 127.5).detach().cpu().numpy()
    
def generate_cf(obs, do_a=None, do_f=None, do_r=None, do_s=None):
    obs = preprocess(norm(obs))
    n_particles = 1 # Number of particles
   
    for k, v in obs.items():
        obs[k] = v.cuda().float()
        if n_particles >= 1:
            ndims = (1,)*3 if k == 'x' else (1,)
            obs[k] = obs[k].repeat(n_particles, *ndims)

    # get counterfactual pa
    do_pa = {}
    with torch.no_grad():
        if do_s != None:
            do_pa['sex'] = torch.tensor(do_s).view(1, 1)
        if do_f != None:
            do_pa['finding'] = torch.tensor(do_f).view(1, 1)
        if do_r != None:
            do_pa['race'] = F.one_hot(torch.tensor(do_r), num_classes=3).view(1, 3)
        if do_a != None:
            do_pa['age'] = torch.tensor(do_a/100*2-1).view(1,1)

    for k, v in do_pa.items():
        do_pa[k] = v.cuda().float().repeat(n_particles, 1)
    
    # generate counterfactual
    out = model.forward(obs, do_pa, cf_particles=8)
    if not 'cfs' in out:
        return np.array([])

    x_cf = postprocess(out['cfs']['x']).mean(1)
    return x_cf

def generate_cfs(dataloader, amount, do_a=None, do_f=None, do_r=None, do_s=None):
    BATCH_SIZE = 1
    count = 0
    cfs = []
    cfs_metrics = []
    for idx, (image, metrics, target) in enumerate(tqdm(dataloader)):
        obs = {'x':image[0][0], 'sex':metrics['sex'][0], 'age':metrics['age'][0], 'race':metrics['race'][0], 'finding':target[0]}
        cf_metrics = {'sex':metrics['sex'][0].item(), 'age':metrics['age'][0].item(),
                      'race':metrics['race'][0].item(), 'finding':target[0].item()}
        
        do_inter = False
        if do_s != None:
            if cf_metrics['sex'] == do_s: continue
            else:
                cf_metrics['sex'] = do_s
                do_inter = True

        if do_f != None:
            if cf_metrics['finding'] == do_f: continue
            else:
                cf_metrics['finding'] = do_f
                do_inter = True

        if do_r != None:
            if cf_metrics['race'] == do_r: continue
            else:
                cf_metrics['race'] = do_r
                do_inter = True

        do_a_post = None
        if do_a != None:
            if (20*do_a<=cf_metrics['age']<=(20*do_a+19)): continue
            else:
                do_a_post = random.randint(do_a*20, do_a*20+19)
                cf_metrics['age'] = do_a_post
                do_inter = True
        
        if do_inter:
            #do_a_post = random.randint(do_a*20, do_a*20+19)
            cf = generate_cf(obs=obs, do_a=do_a_post, do_f=do_f, do_r=do_r, do_s=do_s)
            if len(cf)==0:
                continue

            cfs.append(cf)
            cfs_metrics.append(cf_metrics)

            count += BATCH_SIZE
            if count >= amount:
                return cfs, cfs_metrics, idx

    return cfs, cfs_metrics, -1

def generate_cfs_random(dataloader, amount):
    BATCH_SIZE = 1
    count = 0
    cfs = []
    cfs_metrics = []
    for idx, (image, metrics, target) in enumerate(tqdm(dataloader)):
        obs = {'x':image[0][0], 'sex':metrics['sex'][0], 'age':metrics['age'][0], 'race':metrics['race'][0], 'finding':target[0]}
        cf_metrics = {'sex':metrics['sex'][0].item(), 'age':metrics['age'][0].item(),
                      'race':metrics['race'][0].item(), 'finding':target[0].item()}
        
        cf_metrics['sex'] = random.randint(0,1)
        # cf_metrics['finding'] = random.randint(0,1)
        cf_metrics['race'] = random.randint(0,2)

        do_a = random.randint(0,4)
        cf_metrics['age'] = random.randint(do_a*20, do_a*20+19)
        
        cf = generate_cf(obs=obs, do_a=cf_metrics['age'], do_f=cf_metrics['finding'], do_r=cf_metrics['race'], do_s=cf_metrics['sex'])
        cfs.append(cf)
        cfs_metrics.append(cf_metrics)

        count += BATCH_SIZE
        if count >= amount:
            return cfs, cfs_metrics, idx

    return cfs, cfs_metrics, -1