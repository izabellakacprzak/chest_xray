import sys
import os
sys.path.append(os.path.join(sys.path[0],'pgm_chest'))
import torch
from torch.utils.data import DataLoader
import torchvision
from mimic import MimicDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
from train_setup import setup_directories, setup_tensorboard, setup_logging
# From datasets import get_attr_max_min
from utils import EMA, seed_all
from matplotlib.ticker import LinearLocator
from vae_chest import HVAE
from pgm_chest.layers import TraceStorage_ELBO
from pgm_chest.flow_pgm import FlowPGM
from pgm_chest.dscm import DSCM
from PIL import Image
from matplotlib import colors, patches
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)
rc('image', interpolation='none')
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{fontawesome5} \usepackage{graphics}')
import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
# cwd = os.getcwd()
# print(f"cwd functions {cwd}")

MODELS = {}
DATA = {}

_HEIGHT, _WIDTH = 270, 270
SEX_CAT_CHEST = ['male', 'female']  # 0,1
RACE_CAT = ['White', 'Asian', 'Black']  # 0,1,2
FIND_CAT = ['No disease', 'Pleural Effusion']
finding_categories_ =['No \ disease', 'Pleural \ Effusion']

def load_model_and_data_chest():
    MODELS['chest_dscm'], MODELS['chest_dscm_args'], pgm_args = load_chest_models()
    DATA['test'] = load_chest_data(pgm_args)['test']
    return MODELS['chest_dscm'], MODELS['chest_dscm_args'], DATA['test']

def norm(batch):
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = (batch['x'].float() - 127.5) / 127.5  # [-1,1]
        elif k in ['age']:
            batch[k] = batch[k].float().unsqueeze(-1)
            batch[k] = batch[k] / 100.
            batch[k] = batch[k] *2 -1 #[-1,1]
        elif k in ['race']:
            batch[k] = F.one_hot(batch[k], num_classes=3).squeeze().float()
        elif k in ['finding']:
            batch[k] = batch[k].unsqueeze(-1).float()
        else:
            batch[k] = batch[k].float().unsqueeze(-1)
    return batch

def loginfo(title, logger, stats):
    logger.info(f'{title} | ' +
                ' - '.join(f'{k}: {v:.4f}' for k, v in stats.items()))

def inv_preprocess(pa):
    # Undo [-1,1] parent preprocessing back to original range
    for k, v in pa.items():
        if k =='age':
            pa[k] = (v + 1) / 2 * 100
    return pa


def vae_preprocess(args, pa):
    pa = torch.cat([pa[k] for k in args.parents_x], dim=1)
    pa = pa[..., None, None].repeat(
        1, 1, *(args.input_res,)*2).float()
    return pa

class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

def load_chest_data(args):
    args.bs = 10
    args.data_dir = "src/chest_xray/mimic_subset"
    args.csv_dir =  "src/chest_xray/mimic_subset"
    transf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((args.input_res, args.input_res)),
        ])

    test_set = MimicDataset(root=args.data_dir,
                            csv_file=os.path.join(args.csv_dir, 'mimic.sample.test.csv'),
                            columns = args.parents_x,
                            transform=transf,
                            )

    kwargs = {
            'batch_size': args.bs,
            'num_workers': 8,
            'pin_memory': True,
        }
    dataloaders = {
        'test': DataLoader(test_set, shuffle=False, **kwargs)
    }
    return dataloaders

def load_chest_models():
    # Load predictors
    args = Hparams()
    args.predictor_path = 'src/chest_xray/checkpoints/a_r_s_f/mimic_classifier_resnet18_l2_slurm/checkpoint.pt'
    predictor_checkpoint = torch.load(args.predictor_path)
    args.update(predictor_checkpoint['hparams'])
    predictor = FlowPGM(args)
    
    # Load PGM
    args.pgm_path = 'src/chest_xray/checkpoints/a_r_s_f/sup_pgm_mimic/checkpoint.pt'
    # print(f'\nLoading PGM checkpoint: {args.pgm_path}')
    pgm_checkpoint = torch.load(args.pgm_path)
    pgm_args = Hparams()
    pgm_args.update(pgm_checkpoint['hparams'])
    pgm = FlowPGM(pgm_args)
    # pgm.load_state_dict(pgm_checkpoint['ema_model_state_dict'])

    args.vae_path = 'src/chest_xray/checkpoints/a_r_s_f/mimic_beta9_gelu_dgauss_1_lr3/checkpoint.pt'

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
    args.load_path = f'src/chest_xray/checkpoints/a_r_s_f/{dscm_dir}/{which_checkpoint}.pt'
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

def postprocess(x):
    return ((x + 1.0) * 127.5).detach().cpu().numpy()

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def preprocess(batch):
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = batch['x'].float().cuda()  # [0,1]
        elif k in ['age']:
            batch[k] = batch[k].float().cuda()
        elif k in ['race']:
            batch[k] =batch[k].float().cuda()
        elif k in ['finding']:
            batch[k] = batch[k].float().cuda()
        else:
            batch[k] = batch[k].float().cuda()
    return batch

def get_graph_chest(*args):
    x, a, d, r, s= r'$\mathbf{x}$', r'$a$', r'$d$', r'$r$', r'$s$'
    ua, ud, ur, us = r'$\mathbf{U}_a$', r'$\mathbf{U}_d$', r'$\mathbf{U}_r$', r'$\mathbf{U}_s$'
    zx, ex = r'$\mathbf{z}_{1:L}$', r'$\boldsymbol{\epsilon}$'
    
    G = nx.DiGraph()
    G.add_edge(ua, a)
    G.add_edge(ud, d)
    G.add_edge(ur, r)
    G.add_edge(us, s)
    G.add_edge(a, d)
    G.add_edge(d, x)
    G.add_edge(r, x)
    G.add_edge(s, x)
    G.add_edge(ex, x)
    G.add_edge(zx, x)
    G.add_edge(a, x)

    pos = {
        x: (0,0), 
        a: (-1, 1),
        d: (0, 1),
        r: (1, 1),
        s: (1, 0),
        ua: (-1, 1.75),
        ud: (0, 1.75),
        ur: (1, 1.75),
        us: (1, -0.75),
        zx: (-1, 0),
        ex: (0, -0.75),
    }

    node_c = {}
    for node in G:
        node_c[node] = 'lightgrey' if node in [x, a, d, r, s] else 'white'

    edge_c = {e: 'black' for e in G.edges}
    node_line_c = {k: 'black' for k, _ in node_c.items()}

    if args[0]:  # do_r
        # G.remove_edge(ur, r)
        edge_c[(ur, r)] = 'lightgrey'
        node_line_c[r] = 'red'
    if args[1]:  # do_s
        # G.remove_edges_from([(us, s)])
        edge_c[(us, s)] = 'lightgrey'
        node_line_c[s] = 'red'
    if args[2]:  # do_f (do_d)
        # G.remove_edges_from([(ud, d), (a, d)])
        edge_c[(ud, d)] = 'lightgrey'
        edge_c[(a, d)] = 'lightgrey'
        node_line_c[d] = 'red'
    if args[3]:  # do_a
        # G.remove_edge(ua, a)
        edge_c[(ua, a)] = 'lightgrey'
        node_line_c[a] = 'red'

    fs = 30
    options = {
        "font_size": fs,
        "node_size": 3000,
        "node_color": list(node_c.values()),
        "edgecolors": list(node_line_c.values()),
        "edge_color": list(edge_c.values()),
        "linewidths": 2,
        "width": 2,
    }

    fig, ax = plt.subplots(1, 1, figsize=(5,6))
    # fig.patch.set_visible(False)
    ax.axis("off")
    nx.draw_networkx(G, pos, **options, arrowsize=25, arrowstyle='-|>', ax=ax)
    
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.array(fig.canvas.renderer.buffer_rgba())
    return arr


def resize(x, h=_HEIGHT, w=_WIDTH):
    x = Image.fromarray(x)
    x = x.resize((h, w), resample=Image.Resampling.NEAREST)
    return np.array(x)



def infer_image_chest(*args):
    n_particles = 32 # Number of particles

    idx, _, r, s, f, a = args[:6]
    do_r, do_s, do_f, do_a = args[6:]
    obs = DATA['test'].dataset.__getitem__(int(idx))
    
    for k, v in obs.items():
        obs[k] = v.cuda().float()
        if n_particles > 1:
            ndims = (1,)*3 if k == 'x' else (1,)
            obs[k] = obs[k].repeat(n_particles, *ndims)
    # get founterfactual pa
    do_pa = {}
    with torch.no_grad():
        if do_s:
            do_pa['sex'] = torch.tensor(SEX_CAT_CHEST.index(s)).view(1, 1)
        if do_f:
            do_pa['finding'] = torch.tensor(FIND_CAT.index(f)).view(1, 1)
        if do_r:
            do_pa['race'] = F.one_hot(torch.tensor(RACE_CAT.index(r)), num_classes=3).view(1, 3)
        if do_a:
            do_pa['age'] = torch.tensor(a/100*2-1).view(1,1)
    for k, v in do_pa.items():
        do_pa[k] = v.cuda().float().repeat(n_particles, 1)
    # generate counterfactual 
    out = MODELS['chest_dscm'].forward(obs, do_pa, elbo_fn, cf_particles=1)
    x_cf = postprocess(out['cfs']['x']).mean(0).squeeze()
    _orig_x = postprocess(obs['x']).mean(0).squeeze()
    cf_x = resize(x_cf.astype(np.uint8), h=_HEIGHT, w=_WIDTH)
    de_map = x_cf- _orig_x

    fig, ax = plt.subplots()
    im = ax.imshow(de_map, 
        cmap='RdBu_r', norm=MidpointNormalize(vmin=-255, midpoint=0, vmax=255))
    fig.patch.set_visible(False)
    ax.axis('off')
    cbar = fig.colorbar(im, ticks=[-255, -128, 0, 128, 255], fraction=0.046, pad=0.01, aspect=30);
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.show()
    fig.canvas.draw()
    de_map = np.array(fig.canvas.renderer.buffer_rgba())
    h, w, _ = de_map.shape
    effect = np.array(fig.canvas.renderer.buffer_rgba())
    h, w, _ = effect.shape
    cc = 12
    effect = effect[cc:h-cc, 14:w-10,:]

    # counterfactual uncertainty map
    # plt.close('all')
    fig, ax = plt.subplots()
    cf_x_std = out['cfs']['x'].std(0).squeeze().detach().cpu().numpy()
    # cf_x_std = resize(cf_x_std, h=m_size, w=n_size, resample='bicubic')
    im = ax.imshow(cf_x_std, cmap='jet')
    fig.patch.set_visible(False)
    ax.axis('off')
    cbar = fig.colorbar(im, fraction=0.046, pad=0.01, aspect=30, format="{:.2f}".format);
    cbar.outline.set_visible(False)
    cbar.locator = LinearLocator(numticks=5)
    cbar.update_ticks()
    # cbar.formatter.set_powerlimits((0, 0))
    # ax.ticklabel_format(style='sci')
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    plt.show()
    fig.canvas.draw()
    cf_x_std = np.array(fig.canvas.renderer.buffer_rgba())
    h, w, _ = cf_x_std.shape
    cf_x_std = cf_x_std[cc:h-cc, 14:w-10,:]  # counterfactual uncertainty mapy map


    # counterfactual attributes
    s = SEX_CAT_CHEST[int(obs['sex'].mean(0).clone().cpu().squeeze().numpy())]
    f = FIND_CAT[int(obs['finding'].mean(0).clone().cpu().squeeze().numpy())]
    r = RACE_CAT[obs['race'].mean(0).clone().cpu().squeeze().numpy().argmax(-1)]
    a = (obs['age'].mean(0).clone().cpu().squeeze().numpy()+1)*50

    cf_r = RACE_CAT[out['cf_pa']['race'].mean(0).clone().cpu().squeeze().numpy().argmax(-1)]
    cf_s = SEX_CAT_CHEST[int(out['cf_pa']['sex'].mean(0).clone().cpu().squeeze().numpy())]
    cf_f = FIND_CAT[int(out['cf_pa']['finding'].mean(0).clone().cpu().squeeze().numpy())]
    cf_a = (out['cf_pa']['age'].mean(0).clone().cpu().squeeze().numpy()+1)*50
    return (cf_x, cf_x_std, effect, cf_r, cf_s, cf_f, cf_a)

elbo_fn = TraceStorage_ELBO(num_particles=1)
