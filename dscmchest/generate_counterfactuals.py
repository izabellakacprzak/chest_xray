import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

from dscmchest.functions_for_gradio import load_chest_models

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
    
def generate_cf(obs, do_s=None, do_r=None, do_a=None):
    do_inter = False
    original_metrics = {'sex':obs['sex'].item(), 'age':obs['age'].item(), 'race':obs['race'].item(), 'finding':obs['finding'].item()}
    cf_metrics = original_metrics.copy()
    obs = norm(obs)
    n_particles = 32 # Number of particles
   
    for k, v in obs.items():
        obs[k] = v.cuda().float()
        if n_particles > 1:
            ndims = (1,)*3 if k == 'x' else (1,)
            obs[k] = obs[k].repeat(n_particles, *ndims)
    # get founterfactual pa
    do_pa = {}
    with torch.no_grad():
        if do_s and original_metrics['sex'] != do_s:
            do_inter = True
            do_pa['sex'] = torch.tensor(do_s).view(1, 1)
            cf_metrics['sex'] = do_s
        # if do_f:
        #     do_pa['finding'] = torch.tensor(do_s).view(1, 1)
        if do_r and original_metrics['race'] != do_r:
            do_inter = True
            do_pa['race'] = F.one_hot(torch.tensor(do_r), num_classes=3).view(1, 3)
            cf_metrics['race'] = do_r
        if do_a and original_metrics['age'] != do_a:
            do_inter = True

            # convert age ranges to actual values
            do_a = random.randint(do_a*20, do_a*20+19)
            do_pa['age'] = torch.tensor(do_a/100*2-1).view(1,1)
            cf_metrics['age'] = do_a

    if not do_inter:
       return None, {}

    for k, v in do_pa.items():
        do_pa[k] = v.cuda().float().repeat(n_particles, 1)
        
    # generate counterfactual
    out = model.forward(obs, do_pa, cf_particles=1)
    x_cf = postprocess(out['cfs']['x']).mean(0)
    return x_cf, cf_metrics

def generate_cfs(data, amount, do_s=None, do_a=None, do_r=None):
    count = 0
    cfs = []
    cfs_metrics = []
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    for _, (image, metrics, target) in enumerate(tqdm(dataloader)):
        sample = {'x':image[0][0], 'sex':metrics['sex'][0], 'age':metrics['age'][0], 'race':metrics['race'][0], 'finding':target[0]}
        cf, cf_metrics = generate_cf(obs=sample, do_s=do_s, do_a=do_a, do_r=do_r)
        
        if len(cf_metrics)==0:
            continue

        cfs.append(cf)
        cfs_metrics.append(cf_metrics)

        count += 1
        if count >= amount:
            return cfs, cfs_metrics

    return cfs, cfs_metrics
