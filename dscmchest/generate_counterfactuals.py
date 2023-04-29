import torch
import torch.nn.functional as F

from functions_for_gradio import load_chest_models

model, _, _ = load_chest_models()

def postprocess(x):
    return ((x + 1.0) * 127.5).detach().cpu().numpy()
    
def generate_cf(obs, do_s=None, do_r=None, do_a=None):
    n_particles = 32 # Number of particles
    
    for k, v in obs.items():
        obs[k] = v.cuda().float()
        if n_particles > 1:
            ndims = (1,)*3 if k == 'x' else (1,)
            obs[k] = obs[k].repeat(n_particles, *ndims)
    # get founterfactual pa
    do_pa = {}
    with torch.no_grad():
        if do_s:
            do_pa['sex'] = torch.tensor(do_s).view(1, 1)
        # if do_f:
        #     do_pa['finding'] = torch.tensor(do_s).view(1, 1)
        if do_r:
            do_pa['race'] = F.one_hot(torch.tensor(do_r), num_classes=3).view(1, 3)
        if do_a:
            do_pa['age'] = torch.tensor(do_a/100*2-1).view(1,1)
    for k, v in do_pa.items():
        do_pa[k] = v.cuda().float().repeat(n_particles, 1)
    # generate counterfactual
    out = model.forward(obs, do_pa, cf_particles=1)
    x_cf = postprocess(out['cfs']['x']).mean(0).squeeze()
    print(type(x_cf))
    return x_cf

def generate_cfs(dataloader, do_s, do_a, do_r):
    cfs = []
    for image, metrics, target in dataloader:
        cf = generate_cf(obs=image, do_s=do_s, do_a=do_a, do_r=do_r)
        new_sample = {'x':cf, 'label':target, 'sex':metrics['sex'], 'race':metrics['race'], 'age':metrics['age']}
        cfs.append(new_sample)

    return cfs