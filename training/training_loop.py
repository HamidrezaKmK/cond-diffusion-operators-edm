"""Main training loop."""
from typing import Callable, Dict, Any
import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import wandb

import dnnlib
from training.datasets.dataset import WindowedDataset
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training.noising_kernels import NoisingKernel
from training.augment_pipe import AugmentPipe
from training.collate import Collate
from training.loss import ScoreMatchingLoss

#----------------------------------------------------------------------------

def training_loop(
    run_dir,                       
    # main arguments
    dataset: torch.utils.data.Dataset,
    data_loader_kwargs: Dict[str, Any],
    loss_fn: ScoreMatchingLoss,
    optim_partial: Callable[[torch.nn.Parameter], torch.optim.Optimizer],
    net: torch.nn.Module, 
    augment_pipe: AugmentPipe,
    collate_fn: Collate,
    # TODO: add typehint
    wandb_enabled: bool,
    seed: int,
    base_learning_rate: float,
    batch_size: int,
    batch_gpu: int,
    total_kimg: int,
    ema_halflife_kimg,
    ema_rampup_ratio,
    lr_rampup_kimg,
    loss_scaling,
    kimg_per_tick,
    snapshot_ticks,
    state_dump_ticks,
    resume_pkl,
    resume_state_dump,
    resume_kimg,
    cudnn_benchmark,
    device: torch.device,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset, 
        rank=dist.get_rank(), 
        num_replicas=dist.get_world_size(), 
        seed=seed
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset, 
            sampler=dataset_sampler, 
            batch_size=batch_gpu, 
            collate_fn=collate_fn,
            **data_loader_kwargs,
        )
    )

    # Construct network.
    net.train().requires_grad_(True).to(device)
    
    dist.print0("Number of params: {}".format(misc.count_parameters(net)))

    # Print network statistics.
    if dist.get_rank() == 0:
        with torch.no_grad():
            sample, _ = dataset[0]
            num_coords = sample.shape[0]
            input_coords = torch.zeros([batch_gpu, num_coords, 111], device=device)
            input_values = torch.zeros([batch_gpu, 111], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            conditioning = torch.zeros([batch_gpu, num_coords, 111], device=device)
            misc.print_module_summary(net, [input_coords, input_values, sigma, conditioning], max_nesting=2)


    # Setup optimizer.
    dist.print0('Setting up optimizer...')

    optimizer: torch.optim.Optimizer = optim_partial(net.parameters())
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, 
        device_ids=[device], 
        broadcast_buffers=False,
        find_unused_parameters=True # TODO: this should be commented out!
    )
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
            
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)        
        del data # conserve memory
    if resume_state_dump:
        # Chris: is None by default
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                data, conditioning = next(dataset_iterator)
                coords, samples = data
                coords = coords.to(device).to(torch.float32)
                samples = samples.to(device).to(torch.float32)
                if conditioning is not None:
                    conditioning = conditioning.to(device).to(torch.float32)
                loss = loss_fn(
                    net=ddp, 
                    coords=coords,
                    samples=samples,
                    conditioning=conditioning,
                    augment_pipe=augment_pipe,
                )
                training_stats.report('Loss/loss', loss)
                loss_value = loss.sum().mul(loss_scaling / batch_gpu_total)
                if wandb_enabled:
                    wandb.log({"loss": loss_value.item()})
                loss_value.backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = base_learning_rate * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                training_stats.report("Loss/any_nan", torch.isnan(param.grad).any().item())
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and \
            (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and \
            (done or cur_tick % snapshot_ticks == 0) and \
            (cur_tick != 0):
            data = dict(
                ema=ema, 
                loss_fn=loss_fn, 
                augment_pipe=augment_pipe, 
                dataset_kwargs=dict(dataset_kwargs),
                cur_nimg=cur_nimg # store iteration # here
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                # Save with filename corresponding to iteration.
                #with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                #    pickle.dump(data, f)
                # Chris B: I don't want disk space to scale linearly with time, just
                # save one snapshot. In the future, we can also do them for each
                # validation metric like in DiffusionOperators.
                with open(os.path.join(run_dir, f'network-snapshot.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                
            del data # conserve memory

            # Chris B: write out stats here. I want this to be synchronised with the
            # saving of the checkpoint, so we don't get duplicate entries in the stats
            # file.
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                stats_jsonl.write(
                    json.dumps(
                        dict(training_stats.default_collector.as_dict(), timestamp=time.time())
                    ) + '\n'
                )
                stats_jsonl.flush()
    
        # Save full dump of the training state.
        # Chris B: I don't want to deal with this in my code yet
        #if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
        #    torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.#join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
        # Update logs.
        training_stats.default_collector.update()
        dist.update_progress(cur_nimg // 1000, total_kimg)
        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
