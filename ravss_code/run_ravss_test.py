#!/usr/bin/env python3
import datetime

from pathlib import Path
from collections import defaultdict
import fire
from typing import Dict, List, Any, Union, Sequence, Tuple
import logging

import torch
import numpy as np
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (Engine, Events)
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    CosineAnnealingScheduler,
    global_step_from_engine,
)
import ignite
import ignite.distributed as idist
from ignite.metrics import RunningAverage, Loss, EpochMetric
from ignite.utils import convert_tensor
from torch.cuda.amp import GradScaler, autocast
from ignite.utils import manual_seed, setup_logger
import soundfile as sf

#import dataset
#import dataset_lrs3 as dataset 
#import dataset_target as dataset
import dataset_condition_target_val as dataset
#import dataset_condition_target_val_noise as dataset
import models
import utils
import criterion as losses

import os
import pdb

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def log_metrics(logger, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\n Test time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")

def transfer_to_device(batch):
    DEVICE = idist.device()
    return (x.to(DEVICE, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


def __setup(config: Path,
            default_args=utils.DEFAULT_ARGS,
            **override_kwargs) -> Dict[str, Any]:
    config_parameters = utils.parse_config_or_kwargs(
        config, default_args=default_args, **override_kwargs)
    
    return config_parameters


def log_basic_info(logger, config_parameters):
    logger.info(f"Test {config_parameters['model']} on VoxCeleb2-2Mix")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
    # explicitly import cudnn as
    # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config_parameters.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")

def test(local_rank, config_parameters):
    """Trains a given model specified in the config file or passed as the --model parameter.
    All options in the config file can be overwritten as needed by passing --PARAM
    Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'

    :param config: yaml config file
    :param **kwargs: parameters to overwrite yaml config
    """
    rank = idist.get_rank()
    manual_seed(config_parameters["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name='vox2_ddp')
    

    outputpath = config_parameters['outputpath']
    if rank == 0:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{config_parameters['model']}_test_{now}"
        output_dir = Path(outputpath) / folder_name
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        config_parameters["outputpath"] = output_dir.as_posix()
        log_file = output_dir / 'test.log'
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
        logger.info(f"Output dir: {config_parameters['outputpath']}")
        log_basic_info(logger, config_parameters)

        if "cuda" in device.type:
            config_parameters["cuda device name"] = torch.cuda.get_device_name(local_rank)
  
    test_df = config_parameters['test_scp']
    ds_root = config_parameters['dataset_root_path']
    
    test_ds = dataset.Vox2_Dataset(config_parameters['mix_num'],test_df, ds_root, dstype='test',batch_size = config_parameters['batch_size'])
    nproc = idist.get_nproc_per_node() #batch_size on each node
    test_loader = idist.auto_dataloader(test_ds, batch_size=nproc, num_workers=config_parameters['num_workers'] * nproc, shuffle=False, drop_last=True, collate_fn = dataset.dummy_collate_fn)

    model = getattr(models, config_parameters['model'])(num_spks=config_parameters['mix_num'])
    #model = getattr(models, config_parameters['model'])()
    checkpoint_path = config_parameters.get("test_cdp",None)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path)['model'])
    model = idist.auto_model(model, sync_bn = True)

    sisnr_func = getattr(losses, config_parameters['val_loss'])().to(idist.device())
    loss_func = getattr(losses, config_parameters['loss'])().to(idist.device())
    pesq_func = getattr(losses,config_parameters['pesq'])().to(idist.device())
    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = {
            "si-snr-loss": Loss(sisnr_func),
            "pesq": Loss(pesq_func)
            }

    evaluator = create_evaluator(model, metrics=metrics, pit=loss_func, config_parameters=config_parameters, tag='test')


    try:
        state = evaluator.run(test_loader)
        log_metrics(logger,state.times["COMPLETED"],"Test",state.metrics)
    except Exception as e:
        logger.exception("")
        raise e


def create_evaluator(model, metrics, pit, config_parameters, tag='val'):
    with_amp = config_parameters['with_amp']

    @torch.no_grad()
    def evaluate_step(engine, batch):
        model.eval()
        mixture, source ,condition, _ = batch
        mixture, source, condition = transfer_to_device([mixture, source, condition])
        with autocast(enabled=with_amp):
            pred_wav,compare_a,compare_v = model(mixture, condition,condition.shape[0])

        ############################################################
        # ################# save for visualization ###################
        # num_pred,_ = pred_wav.shape
        # for i in range(num_pred):
        #     sf.write(f'ours_{i}.wav',pred_wav[i].detach().cpu(),16000)
        # ############################################################
        ############################################################
        #for 1-channel
        # num_spks = config_parameters['mix_num']
        # target_num = condition.shape[0]
        # select_source = source
        # _,L = select_source.shape
        # select_source_1 = select_source.reshape(-1,num_spks,L)
        # select_source_no_pit,select_source_pit = select_source_1.split([target_num,num_spks-target_num],dim=1) #[B,t,L]+[B,n-t,L]
        # select_source_pit = select_source_pit.transpose(1,2).contiguous() #(B,L,n-t)
        # _,L2 = pred_wav.shape
        # pred_wav = pred_wav.reshape(-1, num_spks,L)
        # pred_wav_no_pit,pred_wav_pit = pred_wav.split([target_num,num_spks-target_num],dim=1) #[B,t,L]+[B,n-t,L]
        # pred_wav_pit = pred_wav_pit.transpose(1,2).contiguous() #(B,L,2)
        # _,re_pred_wav_pit = pit(select_source_pit,pred_wav_pit,compare_a,compare_v)
        # _,L3,_ = re_pred_wav_pit.shape
        # re_pred_wav_pit = re_pred_wav_pit.transpose(1,2) #(B,t-n,L)
        # re_pred_wav = torch.cat([pred_wav_no_pit,re_pred_wav_pit],dim=1)
        # re_pred_wav = re_pred_wav.reshape(-1,L3).contiguous()

        select_source = source
        re_pred_wav = pred_wav
        ############################################################

        return select_source, re_pred_wav

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    if idist.get_rank() == 0:
        common.ProgressBar(desc=f"Test ({tag})", persist=False).attach(evaluator)
            
    return evaluator

def run(config, **kwargs):
    setup_args = __setup(config, **kwargs)
    config_parameters = setup_args
    config_parameters['master_port']=3333
    spawn_kwargs={"nproc_per_node": config_parameters['nproc_per_node'], "master_port": config_parameters['master_port']}
    with idist.Parallel(backend = config_parameters['backend'], **spawn_kwargs) as parallel:
        parallel.run(test, config_parameters)
   
def get_save_handler(config):
    return DiskSaver(config["outputpath"], require_empty=False)

if __name__ == "__main__":
    fire.Fire({"run": run})