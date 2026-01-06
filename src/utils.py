import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
from rdkit import Chem
import wandb
import pandas as pd
from collections import Counter
import re
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
import os
from rdkit.Chem.rdchem import BondType as BT

def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask

def to_dense_no_edge(x, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    return PlaceHolder(X=X, E=None, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def mask_no_edge(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.X[node_mask == 0] = - 1
        else:
            self.X = self.X * x_mask
        return self



def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except ValueError:
        return 'None'

def smarts_to_smiles(smarts):
    cleaned_smarts = re.sub(r'\[H\]-', '', smarts)
    cleaned_smarts = re.sub(r'-\[H\]', '', cleaned_smarts)
    mol = Chem.MolFromSmarts(cleaned_smarts)
    Chem.SanitizeMol(mol)
    Chem.RemoveStereochemistry(mol)
    smiles = Chem.MolToSmiles(mol)
    return smiles

def smiles2smiles(smiles):
    mol=Chem.MolFromSmiles(smiles)
    mol=Chem.RemoveHs(mol)
    Chem.RemoveStereochemistry(mol)
    smarts = Chem.MolToSmarts(mol)
    if re.search(r'\[H\]', smarts):
        smiles_cleaned=smarts_to_smiles(smarts)
    else:
        smiles_cleaned=Chem.MolToSmiles(mol)
    return smiles_cleaned


def check_substructure_smiles(smiles1, smiles2):
    if smiles1 == 'None' or smiles2 == 'None':
        return False

    else:
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            mol1 = Chem.RemoveHs(mol1)
            mol2 = Chem.RemoveHs(mol2)
            nlen1 = len(mol1.GetSubstructMatches(mol2))
            nlen2 = len(mol2.GetSubstructMatches(mol1))
            if nlen1 >= 1 and nlen2 >= 1:
                return True
            else:
                return False

        except:
           return False

def gen_top_1_by_smiles_list(true_smiles_list, pred_smiles_list):
    top_1_count = 0
    for true_smiles, pred_smiles in zip(true_smiles_list, pred_smiles_list):
        if check_substructure_smiles(true_smiles, pred_smiles):
            top_1_count += 1
    return top_1_count, top_1_count / len(true_smiles_list)


def list_dict_to_list(list_dict):
    grouped_lists = {}
    for d in list_dict:
        for key, value in d.items():
            prefix = key.split('-')[0]
            if prefix not in grouped_lists:
                grouped_lists[prefix] = []
            grouped_lists[prefix].extend(value)
    final_list = [grouped_lists[key] for key in sorted(grouped_lists.keys())]
    return final_list



def gen_top_K_by_smiles_list(true_smiles_list, pred_smiles_list):
    num_runs = len(pred_smiles_list)
    top_1_count = 0
    top_3_count = 0
    top_5_count = 0
    top_10_count = 0
    sum_count = 0

    for i in range(num_runs):
        try:
            assert len(true_smiles_list) == len(pred_smiles_list[i])
        except:
            print(len(true_smiles_list), len(pred_smiles_list[i]))
            print(true_smiles_list)
            print(pred_smiles_list[i])
            raise ValueError

    results_df_list = []
    #print('-'*100)
    #print('test')
    #print(num_runs)
    #print(len(pred_smiles_list))
    #print(pred_smiles_list)
    #print('-'*100)
    for num_run in range(num_runs):
        pred_smiles_list2 = pred_smiles_list[num_run]
        results_df_list.append(pred_smiles_list2)
    results_df = pd.DataFrame(results_df_list).T
    for index, row in results_df.iterrows():
        true_smiles = true_smiles_list[index]
        row_values = row.to_list()
        processed_data = [assert_smiles_valid(smiles) for smiles in row_values]

        counts = Counter(processed_data)

        top_1_smiles_list = [smiles for smiles, count1 in counts.most_common(1)]
        top_3_smiles_list = [smiles for smiles, count3 in counts.most_common(3)]
        top_5_smiles_list = [smiles for smiles, count5 in counts.most_common(5)]
        top_10_smiles_list = [smiles for smiles, count10 in counts.most_common(10)]
        if check_substructure_smiles(true_smiles, top_1_smiles_list[0]):
            top_1_count += 1
        for smiles in top_3_smiles_list:
            if check_substructure_smiles(true_smiles, smiles):
                top_3_count += 1
                break
        for smiles in top_5_smiles_list:
            if check_substructure_smiles(true_smiles, smiles):
                top_5_count += 1
                break
        for smiles in top_10_smiles_list:
            if check_substructure_smiles(true_smiles, smiles):
                top_10_count += 1
                break
        sum_count += 1
    return top_1_count, top_3_count, top_5_count, top_10_count, sum_count

def assert_smiles_valid(smiles):
    if "." in smiles:
        return 'None'
    if 'None' in smiles:
        return 'None'
    try:
        mol = Chem.MolFromSmiles(smiles)
        return smiles
    except:
        return 'None'

def gen_top_K_by_batched_smiles_list(true_smiles_list, pred_smiles_dict):
    num_runs = len(pred_smiles_dict)
    top_1_count = 0
    top_3_count = 0
    top_5_count = 0
    top_10_count = 0
    sum_count = 0

    for k, v in pred_smiles_dict.items():
        try:
            assert len(true_smiles_list) == len(v)
        except Exception as e:
            print(len(true_smiles_list), len(v))
            print(true_smiles_list)
            print(v)
            raise ValueError from e

    results_df_list = []
    for k, v in pred_smiles_dict.items():
        results_df_list.append(v)
    results_df = pd.DataFrame(results_df_list).T
    for index, row in results_df.iterrows():
        true_smiles = true_smiles_list[index]
        pred_values = row.values
        processed_data = np.array([assert_smiles_valid(smiles) for smiles in pred_values])
        processed_data_counter = Counter(processed_data)
        if 'None' in processed_data_counter:
            del processed_data_counter['None']
        if None in processed_data_counter:
            del processed_data_counter[None]
        try:
            top_1_smiles = processed_data_counter.most_common(1)[0][0]
        except:
            top_1_smiles = 'None'
        if check_substructure_smiles(top_1_smiles, true_smiles):
            top_1_count += 1
        num_pred_smiles = len(processed_data_counter)
        if num_pred_smiles >= 3:
            top_3_smiles_list = [processed_data_counter.most_common(3)[i][0] for i in range(3)]
        else:
            top_3_smiles_list = [processed_data_counter.most_common(i+1)[i][0] for i in range(num_pred_smiles)]

        if num_pred_smiles >= 5:
            top_5_smiles_list = [processed_data_counter.most_common(5)[i][0] for i in range(5)]
        else:
            top_5_smiles_list = [processed_data_counter.most_common(i+1)[i][0] for i in range(num_pred_smiles)]
        if num_pred_smiles >= 10:
            top_10_smiles_list = [processed_data_counter.most_common(10)[i][0] for i in range(10)]
        else:
            top_10_smiles_list = [processed_data_counter.most_common(i+1)[i][0] for i in range(num_pred_smiles)]
        for smiles in top_3_smiles_list:
            if check_substructure_smiles(smiles, true_smiles):
                top_3_count += 1
                break
        for smiles in top_5_smiles_list:
            if check_substructure_smiles(smiles, true_smiles):
                top_5_count += 1
                break
        for smiles in top_10_smiles_list:
            if check_substructure_smiles(smiles, true_smiles):
                top_10_count += 1
                break
        sum_count += 1
    return top_1_count, top_3_count, top_5_count, top_10_count, sum_count

def remove_self_loops_matrix(E):
    B, N, _, d = E.size()
    mask = torch.eye(N, dtype=torch.bool, device=E.device)
    mask = mask.unsqueeze(0).unsqueeze(-1)
    mask = mask.expand(B, N, N, d)
    E_no_self_loops = E.clone()
    E_no_self_loops[mask] = 0.0
    return E_no_self_loops

def to_gen_dense_subgraph(E, fea_X, subgraph_mask):
    num_classes = E.shape[-1]
    E = E.argmax(dim=-1)


    subgraph = torch.zeros_like(E)

    C_NMR_X_fea = fea_X[0, :, 0]

    shift2idx = defaultdict(list)
    for idx, shift_value in enumerate(C_NMR_X_fea.tolist()):
        shift2idx[shift_value].append(idx)

    shifts_index_set = set()
    for pair in subgraph_mask:
        shift1, shift2 = pair.tolist()
        if shift1 in shift2idx and shift2 in shift2idx:

            for j_idx in shift2idx[shift1]:
                for k_idx in shift2idx[shift2]:
                    shifts_index_set.add((j_idx, k_idx))

    if shifts_index_set:
        i_idx, j_idx = zip(*shifts_index_set)
        i_idx, j_idx = list(i_idx), list(j_idx)
        subgraph[:, i_idx, j_idx] = num_classes
        subgraph[:, j_idx, i_idx] = num_classes

    subgraph = F.one_hot(subgraph, num_classes=num_classes+1).float()
    subgraph2 = remove_self_loops_matrix(subgraph)

    return subgraph2

def assert_bonds(bond):
    accepted_bonds_HnC_CHn = {BT.SINGLE, BT.DOUBLE, BT.AROMATIC}
    bond_type = bond.GetBondType()
    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()

    if bond_type in accepted_bonds_HnC_CHn:
        if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
            num_hydrogen1 = sum(1 for neighbor in atom1.GetNeighbors() if neighbor.GetSymbol() == 'H')
            num_hydrogen2 = sum(1 for neighbor in atom2.GetNeighbors() if neighbor.GetSymbol() == 'H')
            if num_hydrogen1 > 0 and num_hydrogen2 > 0:
                return True
    return False

def get_num_cosy(mol):
    mol = Chem.AddHs(mol)
    num_cosy = 0
    for bond in mol.GetBonds():
        if assert_bonds(bond):
            num_cosy += 1
    return num_cosy




def draw_smiles_grid(counter, name, MF , COSY , mols_per_row=5, mol_size=(250, 200)):
    valid_mols = []
    valid_counter = Counter()
    print(f'Drawing SMILES grid for {name}...')
    print(f'Molecular Formula: {MF}')
    print(f'Number of COSY: {COSY}')
    #print(counter)
    for smi in counter:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            num_cosy = COSY
            pred_num_cosy = get_num_cosy(mol)

            #print('-' * 100)
            #rint(pred_num_cosy)
            #print(num_cosy)
            #print(rdMolDescriptors.CalcMolFormula(mol))
            #print(MF)
            #print('-' * 100)

            if pred_num_cosy == num_cosy and str(rdMolDescriptors.CalcMolFormula(mol)) == str(MF):
                valid_mols.append(mol)
                valid_counter[smi] += counter[smi]
        except:
            continue

    print(f'Number of unique SMILES: {len(valid_counter)}')
    print(valid_counter)
    if len(valid_mols) == 0:
        print(f'No valid SMILES found for {name}.')
        return

    return valid_counter, name

def draw_smiles_grid2(valid_counter,name,mols_per_row=5, mol_size=(250, 200)):
    print(f'Number of unique SMILES: {len(valid_counter)}')
    print(valid_counter)


    legends = []
    valid_mols = []
    for smi in valid_counter:
        count = valid_counter[smi]
        legends.append(f'{count} / 256')
        mol = Chem.MolFromSmiles(smi)
        valid_mols.append(mol)


    img = Draw.MolsToGridImage(
        valid_mols,
        molsPerRow=mols_per_row,
        subImgSize=mol_size,
        useSVG=False,
        legends=legends,
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, f"{name}.png")
    img.save(img_path)
    print(f"Image saved to {img_path}")











def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'Di_GT_NMR_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

