import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset

from src.datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
import pickle
from rdkit import Chem


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def get_atom_type(atom, types):
    symbol = atom.GetSymbol()
    if symbol == 'C':
        num_hydrogens = sum([1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H'])
        if num_hydrogens == 1:
            return types['CH']
        elif num_hydrogens == 2:
            return types['CH2']
        elif num_hydrogens == 3:
            return types['CH3']
        elif num_hydrogens == 0:
            return types['C']
        else:
            assert False

    elif symbol == 'N':
        num_hydrogens = sum([1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H'])
        if num_hydrogens == 1:
            return types['NH']
        elif num_hydrogens == 2:
            return types['NH2']
        elif num_hydrogens == 0:
            return types['N']
        else:
            print(num_hydrogens)
            assert False

    elif symbol == 'O':
        num_hydrogens = sum([1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H'])
        if num_hydrogens == 1:
            return types['OH']
        elif num_hydrogens == 0:
            return types['O']
        else:
            assert False

    elif symbol == 'H':
        return None

    else:
        assert False


def is_hydrogen_bonded(atom):
    if atom.GetSymbol() != 'H':
        return True

    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C':
            return False
        elif neighbor.GetSymbol() == 'N':
            return False
        elif neighbor.GetSymbol() == 'O':
            return False

    return True


def find_atom_h_bonds(bond):
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()
    if (atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'H') or (
            atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'C'):
        return True

    elif (atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'H') or (
            atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'N'):
        return True

    elif (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'H') or (
            atom1.GetSymbol() == 'H' and atom2.GetSymbol() == 'O'):
        return True
    else:
        return False


def apply_mask_to_edge_index(mask, edge_index):
    new_indices = torch.arange(mask.size(0))[mask]
    index_mapping = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(new_indices)}
    new_edge_index = edge_index.clone()
    for i in range(edge_index.size(1)):
        new_edge_index[0, i] = index_mapping.get(int(edge_index[0, i]), -1)
        new_edge_index[1, i] = index_mapping.get(int(edge_index[1, i]), -1)
    valid_mask = (new_edge_index[0] != -1) & (new_edge_index[1] != -1)
    new_edge_index = new_edge_index[:, valid_mask]
    return new_edge_index


def assign_labels_to_connected_atoms(mol, atom_labels_tensor):
    result_tensor = torch.zeros_like(atom_labels_tensor)
    result_tensor[:, 0] = atom_labels_tensor[:, 0]
    valid_indices = [i for i, (idx, label) in enumerate(atom_labels_tensor) if idx == 1 and label != 0]
    # print(valid_indices)
    labels_sum = {}
    count = {}

    for valid_idx in valid_indices:
        atom_idx = int(valid_idx)
        label = atom_labels_tensor[valid_idx, 1].item()
        # print(label)
        atom = mol.GetAtomWithIdx(atom_idx)
        # mapped_idx = int(get_connected_atom_indices(mol, atom_idx)[0])
        # print(mapped_idx)
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()

            if n_idx not in labels_sum:
                labels_sum[n_idx] = 0
                count[n_idx] = 0

            labels_sum[n_idx] += label
            count[n_idx] += 1
    # print(labels_sum)
    # print(count)

    for n_idx in labels_sum:
        if count[n_idx] > 0:
            avg_label = labels_sum[n_idx] / count[n_idx]
            result_tensor[n_idx, 1] = avg_label

    for valid_idx in valid_indices:
        result_tensor[valid_idx, 1] = 0

    return result_tensor


def assert_bonds(bond):
    accepted_bonds_HnC_CHn = {BT.SINGLE, BT.DOUBLE, BT.AROMATIC}
    #accepted_bonds_CO = {BT.DOUBLE}

    bond_type = bond.GetBondType()
    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()

    # Check HnC-CHn bonds
    if bond_type in accepted_bonds_HnC_CHn:
        if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
            num_hydrogen1 = sum(1 for neighbor in atom1.GetNeighbors() if neighbor.GetSymbol() == 'H')
            num_hydrogen2 = sum(1 for neighbor in atom2.GetNeighbors() if neighbor.GetSymbol() == 'H')
            if num_hydrogen1 > 0 and num_hydrogen2 > 0:
                #if bond_type == BT.SINGLE:
                return 6
    return 0

def assert_bond_is_single_aromatic(bond, mol):
    atom1 = bond.GetBeginAtom()
    atom2 = bond.GetEndAtom()

    atom1_aromatic_count = 0
    atom2_aromatic_count = 0

    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    num_bond_in_rings = 0
    bond_idx = bond.GetIdx()
    for ring in rings:
        if bond_idx in [mol.GetBondBetweenAtoms(ring[i], ring[i + 1] if i + 1 < len(ring) else ring[0]).GetIdx() for i
                          in range(len(ring))]:
            num_bond_in_rings += 1
    for b in atom1.GetBonds():
        if b.GetIsAromatic():
            atom1_aromatic_count += 1

    for b in atom2.GetBonds():
        if b.GetIsAromatic():
            atom2_aromatic_count += 1

    if atom1_aromatic_count == 3 and atom2_aromatic_count == 3 and num_bond_in_rings == 2:
        return True

    return False


class Pcqm4mv2NMRAdvDataset(InMemoryDataset):
    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None, split_seed=None):
        """ stage: train, val, test
            root: data directory
        """
        self.stage = stage
        self.split_seed = split_seed
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.data, self.slices = torch.load(self.processed_paths[self.file_idx])
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx], weights_only=False)
        # self.raw_paths: ['../data/qm9nmr/qm9nmr_pyg/raw/QM9NMR_all_mol_label.pickle']
        # self.processed_paths: ['../data/qm9nmr/qm9nmr_pyg/processed_qm9nmr_train.pt',
        #                        '../data/qm9nmr/qm9nmr_pyg/processed_qm9nmr_val.pt',
        #                        '../data/qm9nmr/qm9nmr_pyg/processed_qm9nmr_test.pt']     ...
        # self.raw_dir: '../data/qm9nmr/qm9nmr_pyg/raw'
        # self.split_paths: ['../data/qm9nmr/qm9nmr_pyg/train.pickle',
        #                    '../data/qm9nmr/qm9nmr_pyg/val.pickle',
        #                    '../data/qm9nmr/qm9nmr_pyg/test.pickle']

    @property
    def raw_file_names(self):
        return "OGB-NMR-f4a-san-10w.pickle"


    @property
    def split_file_name(self):
        return [f'pcqm4mv2nmr_advanced_split_seed{self.split_seed}_train.pickle',
                f'pcqm4mv2nmr_advanced_split_seed{self.split_seed}_val.pickle',
                f'pcqm4mv2nmr_advanced_split_seed{self.split_seed}_test.pickle']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return [f'processed_pcqm4mv2nmr_advanced_split_seed{self.split_seed}_train.pt',
                f'processed_pcqm4mv2nmr_advanced_split_seed{self.split_seed}_val.pt',
                f'processed_pcqm4mv2nmr_advanced_split_seed{self.split_seed}_test.pt']

    def download(self):
        if files_exist(self.split_paths):
            return
        print("Splitting dataset into train, val, and test sets.")
        print("This may take a while.")
        with open(self.raw_paths[0], 'rb') as file:
            dataset = pickle.load(file)

        n_samples = len(dataset)
        n_train = int(n_samples * 0.8)
        n_val = int(n_samples * 0.1)
        n_test = n_samples - n_train - n_val

        train, val, test = np.split(dataset.sample(frac=1, random_state=self.split_seed), [n_train, n_train + n_val])
        train, val, test = train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

        train.to_pickle(self.split_paths[0])
        val.to_pickle(self.split_paths[1])
        test.to_pickle(self.split_paths[2])

    def process(self):
        # mol must with H atoms!
        self.download()
        print("Start processing dataset!")
        RDLogger.DisableLog('rdApp.*')

        types = {'C': 0, 'CH': 1, 'CH2': 2, 'CH3': 3, 'N': 4, 'NH': 5, 'NH2': 6, 'O': 7, 'OH': 8}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3, 'BT-SINGLE-AROMATIC': 4}

        with open(self.split_paths[self.file_idx], 'rb') as file:
            data_df = pickle.load(file)

        data_list = []
        for i in tqdm(range(100)):
        #for i in tqdm(range(len(data_df))):
            ID = data_df.loc[i, 'name']
            mol = data_df.loc[i, 'rdmol']
            smiles = Chem.MolToSmiles(mol)
            C_shifts_dict = data_df.loc[i, '13C'][0]
            H_shifts_dict = data_df.loc[i, 'HC'][0]
            num_atoms = mol.GetNumAtoms()
            label = torch.zeros(num_atoms, 2)

            for k in range(num_atoms):
                atomic_numbers = mol.GetAtomWithIdx(k).GetAtomicNum()
                label[k][0] = atomic_numbers
            for key in C_shifts_dict:
                label[key - 1][1] = C_shifts_dict[key]
            for key in H_shifts_dict:
                label[key - 1][1] = H_shifts_dict[key]

            type_idx = []
            type_idx_mask = []
            for atom in mol.GetAtoms():
                if get_atom_type(atom, types) is not None:
                    type_idx.append(get_atom_type(atom, types))
                type_idx_mask.append(is_hydrogen_bonded(atom))

            row, col, edge_type = [], [], []
            bond_mask = []
            known_bonds_mask = []
            for bond in mol.GetBonds():
                bond_mask.append(find_atom_h_bonds(bond))
                if find_atom_h_bonds(bond) is False:
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    if bond.GetBondType() != BT.AROMATIC:
                        edge_type += 2 * [bonds[bond.GetBondType()] + 1]
                    else:
                        if assert_bond_is_single_aromatic(bond, mol):
                            edge_type += 2 * [bonds['BT-SINGLE-AROMATIC'] + 1]
                        else:
                            edge_type += 2 * [bonds[bond.GetBondType()] + 1]
                    known_bonds_mask.append(assert_bonds(bond))
                    known_bonds_mask.append(assert_bonds(bond))
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)
            edge_index = apply_mask_to_edge_index(torch.tensor(type_idx_mask), edge_index)

            subgraph_mask = F.one_hot(torch.tensor(known_bonds_mask,dtype=torch.long), num_classes=len(bonds)+2).to(torch.float)
            subgraph_mask[:, 0] = 0

            x = torch.tensor(type_idx)
            x_mask = torch.tensor(type_idx_mask)
            x = F.one_hot(x, num_classes=len(types)).float()

            mapping_1H_13C = assign_labels_to_connected_atoms(mol, label)
            H_shifts_tensor = mapping_1H_13C[:, 1].clone().detach()
            H_shifts_tensor = H_shifts_tensor.reshape(-1, 1)
            total_shifts = torch.cat((label, H_shifts_tensor), dim=-1)
            total_shifts = total_shifts[x_mask]
            y = torch.zeros((1, 0), dtype=torch.float)
            x_label = total_shifts[:, 1:].clone().detach()
            zero_indices = torch.where(x_label[:, 0] == 0.0000)[0]
            x_label[zero_indices, 1] = 0.000

            try:
                assert x.size(0) == x_label.size(0)
            except:
                print(x.size(0), x_label.size(0))
                print(ID)
                print(label)
                print(x_label)
                print(x)
                print(edge_index)
                print(edge_attr)
                print(subgraph_mask)
                print(y)
                #print(ID)
                print(smiles)
                print('-----------------')
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, x_label=x_label, smiles=smiles, subgraph_mask=subgraph_mask, ID=ID)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        #assert False
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])



class PCQM4Mv2NMRAdvancedDataModule(AbstractDataModule):
    def __init__(self, cfg, split_seed):
        self.datadir = cfg.dataset.datadir
        transform = None

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': Pcqm4mv2NMRAdvDataset(stage='train', root=root_path,
                                           transform=transform, split_seed=split_seed),
                    'val': Pcqm4mv2NMRAdvDataset(stage='val', root=root_path,
                                         transform=transform, split_seed=split_seed),
                    'test': Pcqm4mv2NMRAdvDataset(stage='test', root=root_path,
                                          transform=transform, split_seed=split_seed)}
        super().__init__(cfg, datasets)


class PCQM4Mv2NMRAdvancedinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, edge_types, recompute_statistics=False):  # datamodule: QM9NMRDataModule
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'pcqm4mv2nmr-advanced'
        self.atom_encoder = {'C': 0, 'CH': 1, 'CH2': 2, 'CH3': 3, 'N': 4, 'NH': 5, 'NH2': 6, 'O': 7, 'OH': 8}
        self.atom_decoder = ['C', 'CH', 'CH2', 'CH3', 'N', 'NH', 'NH2', 'O', 'OH']
        self.valencies = [4, 1, 2, 3, 3, 2, 1, 2, 1]
        self.edge_types = torch.tensor(edge_types)
        # non_edges, single, double, triple, aromatic, must be updated if the dataset is changed

        super().complete_infos()

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            assert False


