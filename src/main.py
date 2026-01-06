import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append('..')
import pathlib
import warnings
import typing
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from omegaconf.nodes import AnyNode
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from omegaconf.base import ContainerMetadata,Metadata
from src import utils
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete, TrainMolecularMetricsDiscrete_pcq
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
from analysis.visualization import MolecularVisualization
from diffusion.extra_features_molecular import ExtraMolecularFeatures

from omegaconf import listconfig

from diffusion.extra_features import NodeCycleFeatures

torch.serialization.add_safe_globals([
    listconfig.ListConfig,
    int,
    NodeCycleFeatures
])


from diffusion.extra_features import KNodeCycles
torch.serialization.add_safe_globals([KNodeCycles])

from diffusion.extra_features import EigenFeatures
torch.serialization.add_safe_globals([EigenFeatures])

from diffusion.extra_features_molecular import ChargeFeature
torch.serialization.add_safe_globals([ChargeFeature])
from diffusion.extra_features_molecular import ValencyFeature
torch.serialization.add_safe_globals([ValencyFeature])

from torch_geometric.data.data import DataEdgeAttr
torch.serialization.add_safe_globals([DataEdgeAttr])

from src.datasets import qm9nmr_augment_dataset
from datasets import pcqm4mv2nmr_advanced_dataset
from datasets import prediction_dataset

safe_classes = [
    Metadata,
    AnyNode,
    ContainerMetadata,
    DictConfig,
    TrainMolecularMetricsDiscrete,
    TrainMolecularMetricsDiscrete_pcq,
    MolecularVisualization,
    ExtraFeatures,
    DummyExtraFeatures,
    ExtraMolecularFeatures,
    qm9nmr_augment_dataset.QM9NMRAuginfos,
    pcqm4mv2nmr_advanced_dataset.PCQM4Mv2NMRAdvancedinfos,
    prediction_dataset.CasePredinfos,
]

torch.serialization.add_safe_globals(safe_classes)
torch.serialization.add_safe_globals([typing.Any])

warnings.filterwarnings("ignore", category=PossibleUserWarning)



def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    gpus = cfg.general.gpus
    number_chain_steps = cfg.general.number_chain_steps
    strategy = cfg.general.strategy
    infer_start_id = cfg.train.infer_start_id
    multi_infer = cfg.train.multi_infer
    visualize_chain = cfg.train.visualize_chain
    batch_size = cfg.train.batch_size
    name = cfg.general.name + '-resume'
    resume = cfg.general.test_only
    dataset_name = cfg.dataset.name

    # 1. Manually load the entire checkpoint with the security check turned OFF.
    print(f"Loading checkpoint with weights_only=False from: {resume}")
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)

    # 2. Manually get the configuration from the checkpoint.
    # In PyTorch Lightning, the config is usually stored in hyper_parameters.
    loaded_cfg = checkpoint['hyper_parameters']['cfg']

    # --- THE FIX IS HERE ---
    # 3. Add the loaded cfg to the model's keyword arguments.
    model_kwargs['cfg'] = loaded_cfg
    # -----------------------
    
    # 4. Create a new, empty model instance with ALL required arguments.
    model = DiscreteDenoisingDiffusion(**model_kwargs)

    # 5. Load the weights (state_dict) from the checkpoint into the new model.
    model.load_state_dict(checkpoint['state_dict'])

    cfg = loaded_cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.gpus = gpus
    cfg.general.number_chain_steps = number_chain_steps
    cfg.general.strategy = strategy
    cfg.train.infer_start_id = infer_start_id
    cfg.train.multi_infer = multi_infer
    cfg.train.visualize_chain = visualize_chain
    cfg.train.batch_size = batch_size
    cfg.dataset.name = dataset_name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model

def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)

    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '-resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    datasets_split_seed = cfg.dataset.split_seed
    datasets_edge_types = cfg.dataset.edge_types
    print(dataset_config["name"])

    if 'qm9nmr-augment' in dataset_config["name"] and 'prediction' not in dataset_config["name"]:
        from src.datasets import qm9nmr_augment_dataset
        datamodule = qm9nmr_augment_dataset.QM9NMRAugDataModule(cfg, split_seed=datasets_split_seed)
        dataset_infos = qm9nmr_augment_dataset.QM9NMRAuginfos(datamodule=datamodule,
                                                              edge_types=datasets_edge_types,
                                                              recompute_statistics=False)
        train_metrics = TrainMolecularMetricsDiscrete()
    elif 'pcqm4mv2nmr-advanced' in dataset_config["name"] and 'prediction' not in dataset_config["name"]:
        from datasets import pcqm4mv2nmr_advanced_dataset
        datamodule = pcqm4mv2nmr_advanced_dataset.PCQM4Mv2NMRAdvancedDataModule(cfg, split_seed=datasets_split_seed)
        dataset_infos = pcqm4mv2nmr_advanced_dataset.PCQM4Mv2NMRAdvancedinfos(datamodule=datamodule,
                                                                                     edge_types=datasets_edge_types,
                                                                             recompute_statistics=False)
        train_metrics = TrainMolecularMetricsDiscrete_pcq()
    elif 'prediction' in dataset_config["name"]:
        from datasets import prediction_dataset
        datamodule = prediction_dataset.CasePredDataModule(cfg)
        dataset_infos = prediction_dataset.CasePredinfos(cfg)
        train_metrics = TrainMolecularMetricsDiscrete_pcq()
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features)
    else:
        extra_features = DummyExtraFeatures()
        

    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    if 'prediction' not in dataset_config["name"]:
        dataset_infos.compute_input_output_dims(dataset_config["name"], datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
        #print("Input dims:", dataset_infos.input_dims)
        #print("Output dims:", dataset_infos.output_dims)
        #assert False
    else:
        if 'qm9nmr-augment' in dataset_config["name"]:
            dataset_infos.input_dims = {'X': 19, 'E': 11, 'y': 11}
            dataset_infos.output_dims = {'X': 9, 'E': 5, 'y': 0}

        elif 'pcqm4mv2nmr-advanced' in dataset_config["name"]:
            dataset_infos.input_dims = {'X': 19, 'E': 13, 'y': 11}
            dataset_infos.output_dims = {'X': 9, 'E': 6, 'y': 0}

        else:
            raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))
    print("Input dims:", dataset_infos.input_dims)
    print("Output dims:", dataset_infos.output_dims)


    visualization_tools = MolecularVisualization(dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}


    if 'prediction' not in dataset_config["name"]:
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = None
    if cfg.general.test_only:
        cfg, model = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        cfg, model = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])



    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    name = cfg.general.name

    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    use_gpu = torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy=cfg.general.strategy,   # "auto","ddp",None ,"ddp_find_unused_parameters_true"
                      accelerator='gpu' if use_gpu else 'cpu',
                      #devices=cfg.general.gpus,# [2],[3],[4]
                      devices=cfg.general.gpus if use_gpu else [0],  # [2],[3],[4]
                      #devices=[0], #[3],[4]
                      #devices=cfg.general.device,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger=[])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()