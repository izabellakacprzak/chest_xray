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
    
def generate_cf(obs, do_a=None, do_f=None, do_r=None, do_s=None):
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
        if do_s != None:
            do_pa['sex'] = torch.tensor(do_s).view(1, 1)
        if do_f != None:
            do_pa['finding'] = torch.tensor(do_f).view(1, 1)
        if do_r != None:
            do_pa['race'] = F.one_hot(torch.tensor(do_r), num_classes=3).view(1, 3)
        if do_a != None:
            # convert age ranges to actual values
            do_a = random.randint(do_a*20, do_a*20+19)
            do_pa['age'] = torch.tensor(do_a/100*2-1).view(1,1)

    for k, v in do_pa.items():
        do_pa[k] = v.cuda().float().repeat(n_particles, 1)

    # generate counterfactual
    out = model.forward(obs, do_pa, cf_particles=1)
    x_cf = postprocess(out['cfs']['x']).mean(0)
    return x_cf

def generate_cfs(data, amount, do_a=None, do_f=None, do_r=None, do_s=None):
    BATCH_SIZE = 32
    count = 0
    cfs = []
    cfs_metrics = []
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
    for _, (image, metrics, target) in enumerate(tqdm(dataloader)):
        obs = {'x':image[0], 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race'], 'finding':target}
        cf = generate_cf(obs=obs, do_a=do_a, do_f=do_f, do_r=do_r, do_s=do_s)

        cfs.append(cf)
        cf_metrics = metrics.copy()
        if do_s != None:
            cf_metrics['sex'] = [do_s for _ in range(BATCH_SIZE)]
        if do_f != None:
            cf_metrics['finding'] = [do_f for _ in range(BATCH_SIZE)]
        if do_r != None:
            cf_metrics['race'] = [do_r for _ in range(BATCH_SIZE)]
        if do_a != None:
            cf_metrics['age'] = [do_a for _ in range(BATCH_SIZE)]

        cfs_metrics.append(cf_metrics)

        count += BATCH_SIZE
        if count >= amount:
            return cfs, cfs_metrics

    return cfs, cfs_metrics
