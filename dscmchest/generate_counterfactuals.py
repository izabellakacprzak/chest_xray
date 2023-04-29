import torch
import torch.nn.functional as F

from dscmchest.functions_for_gradio import load_chest_models

model, _, _ = load_chest_models()

def postprocess(x):
    return ((x + 1.0) * 127.5).detach().cpu().numpy()
    
def generate_cf(obs, do_s=None, do_r=None, do_a=None):
    original_metrics = {'sex':obs['sex'], 'age':obs['age'], 'race':obs['race'], 'label':obs['label']}
    cf_metrics = {'sex':obs['sex'], 'age':obs['age'], 'race':obs['race'], 'label':obs['label']}

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
            do_pa['sex'] = torch.tensor(do_s).view(1, 1)
            cf_metrics['sex'] = do_s
        # if do_f:
        #     do_pa['finding'] = torch.tensor(do_s).view(1, 1)
        if do_r and original_metrics['race'] != do_r:
            do_pa['race'] = F.one_hot(torch.tensor(do_r), num_classes=3).view(1, 3)
            cf_metrics['race'] = do_r
        if do_a and original_metrics['age'] != do_a:
            do_pa['age'] = torch.tensor(do_a/100*2-1).view(1,1)
            cf_metrics['age'] = do_a
    for k, v in do_pa.items():
        do_pa[k] = v.cuda().float().repeat(n_particles, 1)
    # generate counterfactual
    out = model.forward(obs, do_pa, cf_particles=1)
    x_cf = postprocess(out['cfs']['x']).mean(0).squeeze()
    print(type(x_cf))
    return x_cf, cf_metrics

def generate_cfs(data, do_s=None, do_a=None, do_r=None):
    cfs = []
    cfs_metrics = []
    for sample in data:
        cf, cf_metrics = generate_cf(obs=sample, do_s=do_s, do_a=do_a, do_r=do_r)
        cfs.append(cf)
        cfs_metrics.append(cf_metrics)

    return cfs, cfs_metrics