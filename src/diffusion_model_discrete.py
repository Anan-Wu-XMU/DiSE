import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import time
import os

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
from tqdm import tqdm
from datetime import datetime
from collections import Counter

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'marginal':
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            x_limit = None
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)
        else:
            raise NotImplementedError("Only marginal transition is supported for now.")

        self.save_hyperparameters(ignore=['train_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.test_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.num_test_molecules = 0
        self.num_test_molecules_valid = 0
        self.num_test_molecules_valid_top_1 = 0
        self.num_test_batches = 0
        self.pred_smiles_list_keep = []
        self.true_smiles_list_keep = []
        self.pred_smiles_list_each_run = []
        self.MF = []
        self.num_cosy = 0

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        dense_data_fea, node_mask_fea = utils.to_dense(data.x_label, data.edge_index, data.edge_attr, data.batch)
        dense_data_fea = dense_data_fea.mask(node_mask_fea)

        dense_data_subgraph, node_mask_subgraph = utils.to_dense(data.x, data.edge_index, data.subgraph_mask, data.batch)
        dense_data_subgraph = dense_data_subgraph.mask(node_mask_subgraph)
        subgraph_E = dense_data_subgraph.E
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask, dense_data_fea.X, subgraph_E)
        loss = self.train_loss(masked_pred_E=pred.E, pred_y=pred.y,
                               true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_E=pred.E,true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        self.print("Size of the output features", self.Xdim_output, self.Edim_output, self.ydim_output)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}:"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}:{epoch_bond_metrics}")
        #print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_E_kl.reset()
        self.val_E_logp.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        dense_data_fea, node_mask_fea = utils.to_dense(data.x_label, data.edge_index, data.edge_attr, data.batch)
        dense_data_fea = dense_data_fea.mask(node_mask_fea)

        dense_data_subgraph, node_mask_subgraph = utils.to_dense(data.x, data.edge_index, data.subgraph_mask, data.batch)
        dense_data_subgraph = dense_data_subgraph.mask(node_mask_subgraph)
        subgraph_E = dense_data_subgraph.E

        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, dense_data_fea.X, subgraph_E)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, dense_data_fea.X, subgraph_E, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_E_kl.compute() * self.T, self.val_E_logp.compute()]

        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/E_kl": metrics[1],
                       "val/E_logp": metrics[2]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f}", f"Val Edge type KL: {metrics[1] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_E_kl.reset()
        self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        #####if 'prediction' in self.name:
        if 'pcqm4mv2nmr' in self.name:
            pred_tensor_dict = {}

            print('Starting custom CASE test step !!!!!!!')
            print('Please wait')
            torch.set_printoptions(profile="full")
            MF = data.MF
            self.MF.append(MF)
            self.num_cosy = data.num_cosy
            pred_mol_list = []
            pred_smiles_list = []
            pred_id_list = []
            molecule_list = [] # for visualization
            dense_data, node_mask = utils.to_dense_no_edge(data.x, data.batch)
            dense_data = dense_data.mask_no_edge(node_mask)
            dense_data_fea, node_mask_fea = utils.to_dense_no_edge(data.x_label, data.batch)
            dense_data_fea = dense_data_fea.mask_no_edge(node_mask_fea)
            z_T = diffusion_utils.sample_discrete_feature_noise(dense_data.X, limit_dist=self.limit_dist,
                                                            node_mask=node_mask)
            X, E, y = z_T.X, z_T.E, z_T.y
            
 
            ########
            X1 = X.clone().cpu()
            E1 = E.clone().cpu()

            X1_ = X1.argmax(dim=-1)
            E1_ = E1.argmax(dim=-1)

            #print(f'X1{X1_}')
            #print(f'E1{E1_}')
            #print('111111111111')
            pred_tensor_dict['X_init'] = X1_
            pred_tensor_dict['E_init'] = E1_
            #######



            assert (E == torch.transpose(E, 1, 2)).all()
            dense_data_subgraph = utils.to_gen_dense_subgraph(E, dense_data_fea.X, data.subgraph_mask)
            subgraph_E = dense_data_subgraph
            #torch.set_printoptions(profile="full")
            #print(z_T.X)
            #print(z_T.X.argmax(dim=-1))
            #print('-'*10)
            #print(z_T.E)
            #print(z_T.E.argmax(dim=-1))
            #print('-'*10)
            #print(subgraph_E)
            #print(subgraph_E.argmax(dim=-1))
            #print('-'*10)
            #print(dense_data_fea.X)
            #assert False
            number_chain_steps = self.number_chain_steps
            keep_chain = data.y.shape[0]  # actual number of batch size,
            assert number_chain_steps < self.T

            chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
            chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))
            chain_X = torch.zeros(chain_X_size)
            chain_E = torch.zeros(chain_E_size)
            self.print('Starting to reconstruct the Markov chain, this step is quite time-consuming...')

            for s_int in tqdm(reversed(range(0, self.T))):
                s_array = s_int * torch.ones((data.y.shape[0], 1)).type_as(y)
                t_array = s_array + 1
                s_norm = s_array / self.T
                t_norm = t_array / self.T

                ####
                E2 = E.clone().cpu()
                E2_ = E2.argmax(dim=-1)
                #print(f'E2{E2_}')
                #print(s_int)
                #print('222222222222')

                pred_tensor_dict[f'E_{s_int}_in'] = E2_

                ####
                # Sample z_s
                sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask,
                                                                          dense_data_fea.X, subgraph_E)
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y


                ####
                E3 = E.clone().cpu()
                E3_ = E3.argmax(dim=-1)
                #print(f'E3{E3_}')
                #print('333333333333')
                pred_tensor_dict[f'E_{s_int}'] = E3_


                # Save the first keep_chain graphs
                write_index = (s_int * number_chain_steps) // self.T
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

            #import pickle
            #with open('pred_tensor_dict_vv2.pickle', 'wb') as f:
            #    pickle.dump(pred_tensor_dict, f)

            # Sample
            sampled_s = sampled_s.mask(node_mask, collapse=True)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Prepare the chain for saving
            if keep_chain > 0:
                final_X_chain = X[:keep_chain]
                final_E_chain = E[:keep_chain]

                chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
                chain_E[0] = final_E_chain

                chain_X = diffusion_utils.reverse_tensor(chain_X)
                chain_E = diffusion_utils.reverse_tensor(chain_E)

                # Repeat last frame to see final sample better
                chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
                chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
                assert chain_X.size(0) == (number_chain_steps + 10)
            for i in tqdm(range(data.y.shape[0])):
                atom_types = X[i, :].cpu()
                edge_types = E[i, :, :].cpu()
                molecule_list.append([atom_types, edge_types]) # for visualization
                mol = self.visualization_tools.mol_from_graphs(atom_types.numpy(), edge_types.numpy())
                pred_mol_list.append(mol)
                smiles = utils.mol2smiles(mol)
                pred_smiles_list.append(smiles)
                pred_id_list.append(data.ID[i])
                self.pred_smiles_list_each_run.append(smiles)
                #print(self.cfg.train.visualize_chain)
                #assert False
            if self.cfg.train.visualize_chain:
                self.print('Visualizing chains...')
                num_molecules = chain_X.size(1)  # number of molecules
                #path1 = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
                path1 = os.path.dirname(os.path.dirname(os.getcwd()))
                path2 = os.path.join(path1, 'inference-results')
                if not os.path.exists(path2):
                    os.makedirs(path2)
                current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                this_run_path = os.path.join(path2, f'{current_time}-{self.cfg.general.name}')
                if not os.path.exists(this_run_path):
                    os.makedirs(this_run_path)
                # Visualize the final molecules
                self.visualization_tools.visualize(this_run_path, molecule_list, data.y.shape[0])
                self.print("Visualizing molecules...")
                for j in tqdm(range(num_molecules)):
                    results_path = os.path.join(this_run_path,
                                            f'epoch-{self.current_epoch}-'
                                            f'batches-{self.num_test_batches}-id-{data.ID[j].item()}-{j}-chains')
                    print(f"Saving results to {results_path}")

                    if not os.path.exists(results_path):
                        os.makedirs(results_path, exist_ok=True)
                    _ = self.visualization_tools.visualize_chain(results_path,
                                                                 chain_X[:, j, :].numpy(),
                                                                 chain_E[:, j, :].numpy())
                    self.print('\r{}/{} complete'.format(j + 1, num_molecules), end='', flush=True)
                self.print('\nVisualizing molecules...')
            print('Done generating molecules...')
        else:
            print('Starting test step...,please wait')
            true_smiles_list = []
            for i in tqdm(range(data.y.shape[0])):
                true_smiles = data.smiles[i]
                true_smiles = utils.smiles2smiles(true_smiles)
                true_smiles_list.append(true_smiles)
                self.true_smiles_list_keep.append(true_smiles)
            print('Done generating true molecules...')
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            dense_data_fea, node_mask_fea = utils.to_dense(data.x_label, data.edge_index, data.edge_attr, data.batch)
            dense_data_fea = dense_data_fea.mask(node_mask_fea)

            dense_data_subgraph, node_mask_subgraph = utils.to_dense(data.x, data.edge_index, data.subgraph_mask,
                                                                         data.batch)
            dense_data_subgraph = dense_data_subgraph.mask(node_mask_subgraph)
            subgraph_E = dense_data_subgraph.E
            #torch.set_printoptions(profile="full")
            #print(subgraph_E[0,:])
            #assert False
            # Sample noise  -- z has size (n_samples, n_nodes, n_features)
            number_chain_steps = self.number_chain_steps  # default 50
            assert number_chain_steps < self.T
            infer_start_id = self.cfg.train.infer_start_id
            multi_infer = self.cfg.train.multi_infer
            n_run_smiles_dict = {}
            n_run_mol_dict = {}
            n_run_initE_dict = {}
            for run in range(infer_start_id, infer_start_id + multi_infer):
                pred_mol_list = []
                pred_smiles_list = []
                z_T = diffusion_utils.sample_discrete_feature_noise(dense_data.X, limit_dist=self.limit_dist,
                                                                    node_mask=node_mask)
                X, E, y = z_T.X, z_T.E, z_T.y # (bs, n, d_x_in), (bs, n, n, d_e_int), (bs, n, d_y)
                E_init = E.clone()
                E_init = E_init.argmax(dim=-1).cpu().numpy()

                assert (E == torch.transpose(E, 1, 2)).all()
                self.print(f'Starting to reconstruct the Markov chain in run {run} test batch {self.num_test_batches}, '
                           f'this step is quite time-consuming...')
                # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
                for s_int in tqdm(reversed(range(0, self.T))):
                    s_array = s_int * torch.ones((data.y.shape[0], 1)).type_as(y)
                    t_array = s_array + 1
                    s_norm = s_array / self.T
                    t_norm = t_array / self.T
                    # Sample z_s
                    sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask,
                                                                          dense_data_fea.X, subgraph_E)
                    X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

                # Sample
                sampled_s = sampled_s.mask(node_mask, collapse=True)
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
                for i in tqdm(range(data.y.shape[0])):
                    atom_types = X[i, :].cpu()
                    edge_types = E[i, :, :].cpu()
                    mol = self.visualization_tools.mol_from_graphs(atom_types.numpy(), edge_types.numpy())
                    pred_mol_list.append(mol)
                    smiles = utils.mol2smiles(mol)
                    pred_smiles_list.append(smiles)
                n_run_smiles_dict[f'{run}-{self.num_test_batches}']=pred_smiles_list
                n_run_mol_dict[f'{run}-{self.num_test_batches}']=pred_mol_list
                n_run_initE_dict[f'{run}-{self.num_test_batches}']=E_init
            self.pred_smiles_list_keep.append(n_run_smiles_dict)
            self.num_test_molecules += len(true_smiles_list)
            self.num_test_batches += 1
            Top_1_count, Top_3_count, Top_5_count, Top_10_count, Sum_count = utils.gen_top_K_by_batched_smiles_list(true_smiles_list, n_run_smiles_dict)
            print('This batch has {} molecules, and run number is {}'.format(len(true_smiles_list), multi_infer))
            print('In test batch{}, we have {} top 1 molecules'.format(self.num_test_batches, Top_1_count))
            print('In test batch{}, we have {} top 3 molecules'.format(self.num_test_batches, Top_3_count))
            print('In test batch{}, we have {} top 5 molecules'.format(self.num_test_batches, Top_5_count))
            print('In test batch{}, we have {} top 10 molecules'.format(self.num_test_batches, Top_10_count))
            print('In test batch{}, Top 1 accuracy is {:.3f}'.format(self.num_test_batches, Top_1_count/Sum_count))
            print('In test batch{}, Top 3 accuracy is {:.3f}'.format(self.num_test_batches, Top_3_count/Sum_count))
            print('In test batch{}, Top 5 accuracy is {:.3f}'.format(self.num_test_batches, Top_5_count/Sum_count))
            print('In test batch{}, Top 10 accuracy is {:.3f}'.format(self.num_test_batches, Top_10_count/Sum_count))
            print('Done generating molecules...')

        return {'loss': 0}

    def on_test_epoch_end(self) -> None:
        #if 'prediction' not in self.name:
        #    print('-'*150)
        #    print('Starting the test epoch end evaluation...')
        #    self.print(f'The number of true smiles: {len(self.true_smiles_list_keep)}')
        #    name = self.cfg.general.name
        #    multi_infer = self.cfg.train.multi_infer
        #    infer_start_id = self.cfg.train.infer_start_id
        #    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            # save the true and pred smiles list to txt files
        #    if not os.path.exists(f'{timestamp}-{name}-true-smiles.txt'):
        #        with open(f'{timestamp}-{name}-true-smiles.txt', 'w') as f:
        #            for item in self.true_smiles_list_keep:
        #                f.write(f"{item}\n")
        #        absolute_path = os.path.abspath(f'{timestamp}-{name}-true-smiles.txt')
        #        print('-' * 150)
        #        print(f"Successfully saved true smiles to {absolute_path}")
        #    pred_smiles_results = utils.list_dict_to_list(self.pred_smiles_list_keep)
        #    assert len(pred_smiles_results) == multi_infer
        #    for run_number in range(infer_start_id, infer_start_id + multi_infer):
        #        self.print(f'Run {run_number} has {len(pred_smiles_results[run_number - infer_start_id])} predicted smiles')
        #        with open(f"{timestamp}-{name}-pred-smiles-{run_number}.txt", 'w') as f:
        #            for item in pred_smiles_results[run_number - infer_start_id]:
        #                f.write(f"{item}\n")
        #            absolute_path = os.path.abspath(f'{name}-pred-smiles-{run_number}.txt')
        #            print('-'*150)
        #            top_1_count, top_1_acc = utils.gen_top_1_by_smiles_list(self.true_smiles_list_keep, pred_smiles_results[run_number - infer_start_id])
        #            print(f"Run {run_number} has {top_1_count} top 1 predicted smiles")
        #            self.print(f"Run {run_number} has {self.num_test_molecules} molecules")
        #            print(f"Run {run_number} has {top_1_acc:.3f} top 1 accuracy")
        #            print(f"Successfully saved pred smiles to {absolute_path}")
        #            print('-'*150)

        #    self.print(f"Successfully saved true and pred smiles files...")
            # multi top K accuracy
        #    (top_1_count,
        #     top_3_count,
        #     top_5_count,
        #     top_10_count,
        #     sum_count) = utils.gen_top_K_by_smiles_list(self.true_smiles_list_keep,
        #                                                        pred_smiles_results)

        #   top_1_acc = top_1_count / sum_count
        #    top_3_acc = top_3_count / sum_count
        #    top_5_acc = top_5_count / sum_count
        #    top_10_acc = top_10_count / sum_count

        #     print(f"Sum count: {sum_count}")
        #    print(f"Number of top 1 molecules: {top_1_count}")
        #    print(f"Number of top 3 molecules: {top_3_count}")
        #    print(f"Number of top 5 molecules: {top_5_count}")
        #    print(f"Number of top 10 molecules: {top_10_count}")

        #    print(f"Top 1 accuracy: {top_1_acc:.3f}")
        #    print(f"Top 3 accuracy: {top_3_acc:.3f}")
        #    print(f"Top 5 accuracy: {top_5_acc:.3f}")
        #    print(f"Top 10 accuracy: {top_10_acc:.3f}")
        #    print('Done!')
        #else:
        self.print('Starting the calculation of SMILES distribution statistics...')
        self.pred_smiles_list_each_run = [utils.assert_smiles_valid(smiles) for smiles in self.pred_smiles_list_each_run]
        counter = Counter(self.pred_smiles_list_each_run)
        if 'None' in counter:
            del counter['None']
        molecular_formula = self.MF[0][0]
        HH_COSY = self.num_cosy[0].item()

        valid_counter, name=utils.draw_smiles_grid(counter, name = self.cfg.general.name, MF= molecular_formula,COSY=HH_COSY)
        utils.draw_smiles_grid2(valid_counter, name)
        print('Done!')

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        bs, n, _ = X.shape

        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_E, probE = diffusion_utils.mask_distributions(true_E=limit_E.clone(),
                                                                 pred_E=probE,
                                                                 node_mask=node_mask)
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=None, E=E, y=y, X_t=None, E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=None, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=None, E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_E, prob_pred.E = diffusion_utils.mask_distributions(true_E=prob_true.E,
                                                                      pred_E=prob_pred.E,
                                                                      node_mask=node_mask)
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * kl_e

    def reconstruction_logp(self, t, X, E, node_mask, X_fea, E_subgraph):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(X=X, probE=probE0, node_mask=node_mask)

        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert E.shape == E0.shape

        sampled_0 = utils.PlaceHolder(X=X, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask, X_fea, E_subgraph)

        # Normalize predictions
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)
        return utils.PlaceHolder(X=X, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(X=X, probE=probE, node_mask=node_mask)

        X_t = sampled_t.X
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}

        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, X_fea, E_subgraph,test=False):
        t = noisy_data['t']

        # The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask, X_fea, E_subgraph)

        loss_term_0 = self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask, X_fea, E_subgraph):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        X = torch.cat((X, X_fea), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        if 'advanced' in self.cfg.general.name or 'augment' in self.cfg.general.name:
            E = torch.cat((E, E_subgraph), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()

        return self.model(X, E, y, node_mask)

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, X_fea, E_subgraph):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, X_fea, E_subgraph)

        # Normalize predictions
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(X_t, prob_E, node_mask=node_mask)

        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert E_t.shape == E_s.shape

        out_one_hot = utils.PlaceHolder(X=X_t, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_t, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)
        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)