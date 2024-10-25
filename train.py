"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import jstyleson as json
import glob
import torch
import dnnlib
import pickle

from random_words import RandomWords
import wandb
import dotenv
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import warnings

from torch_utils import distributed as dist
from training import training_loop

warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

@hydra.main(version_base=None, config_path="conf/", config_name="train")
def main(cfg: DictConfig):
    """
    Train diffusion-based generative model based on the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    random_word_generator = RandomWords()
    run_name = f"{random_word_generator.random_word()}_{random_word_generator.random_word()}"
    outdir = os.path.join(os.getenv('SAVE_DIR', 'outputs/exps'), run_name)
    
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True), 
            name=run_name,
            tags=[f"{key}:{value}" for key, value in cfg.wandb.tags.items()] if "tags" in cfg.wandb else None,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
        )
    
    dset = instantiate(cfg.dataset)
    augment_pipe = instantiate(cfg.augment_pipe)
    optim_partial = instantiate(cfg.optimizer)
    score_net = instantiate(cfg.score_net)
    preconditioned_net = instantiate(cfg.preconditioner)(score_net=score_net)
    sampler = instantiate(cfg.sampler)
    loss_fn = instantiate(cfg.loss)(sampler=sampler)

    # Random seed.
    if cfg.seed is None:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        cfg.seed = int(seed)

    if cfg.get('resume', False):
        # Find all the network snapshot files
        snapshots = sorted(
            glob.glob("{}/network-snapshot.pkl".format(outdir))
        )
        if len(snapshots) != 0:
            latest_snapshot = snapshots[-1]
            dist.print0("Found snapshot: {} ...".format(latest_snapshot))
            cfg.resume_pkl = latest_snapshot
            # HACK: we actually have to open the pkl here to
            # get the epoch number.
            with dnnlib.util.open_url(cfg.resume_pkl, verbose=(dist.get_rank() == 0)) as f:
                cfg.resume_kimg = pickle.load(f)['cur_nimg'] // 1000
        cfg.resume_state_dump = None    


    # Create output directory.
    dist.print0('Creating output directory...')
    # Pick output directory.
    if dist.get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'training_options.json'), 'wt') as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=4)
        dnnlib.util.Logger(file_name=os.path.join(outdir, 'log.txt'), file_mode='a', should_flush=True)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0()
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0()

    # Dry run?
    if cfg.get('dry_run', False):
        dist.print0('Dry run; exiting.')
        return
    
    training_loop.training_loop(
        # saving directory
        run_dir=outdir,
        # main arguments
        dataset=dset,
        data_loader_kwargs=OmegaConf.to_container(cfg.dataloader_args),
        loss_fn=loss_fn,
        optim_partial=optim_partial,
        sampler=sampler,
        net=preconditioned_net,
        augment_pipe=augment_pipe,
        # 
        seed=cfg.seed,
        base_learning_rate=cfg.base_learning_rate,
        batch_size=cfg.batch_size,
        batch_gpu=cfg.batch_gpu,
        total_kimg=cfg.total_kimg,
        ema_halflife_kimg=cfg.ema_halflife_kimg,
        ema_rampup_ratio=cfg.ema_rampup_ratio,
        lr_rampup_kimg=cfg.lr_rampup_kimg,
        loss_scaling=cfg.loss_scaling,
        kimg_per_tick=cfg.kimg_per_tick,
        snapshot_ticks=cfg.snapshot_ticks,
        state_dump_ticks=cfg.state_dump_ticks,
        resume_pkl=cfg.resume_pkl,
        resume_state_dump=cfg.resume_state_dump,
        resume_kimg=cfg.resume_kimg,
        cudnn_benchmark=cfg.cudnn_benchmark,
        device=torch.device(cfg.device),
    )

if __name__ == "__main__":
    
    dotenv.load_dotenv(override=True)
    main()

#----------------------------------------------------------------------------
