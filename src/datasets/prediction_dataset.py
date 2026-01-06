import os
import os.path as osp
import pathlib
from pathlib import Path
import torch
import torch.nn.functional as F
from rdkit import RDLogger
from torch_geometric.data import Data, InMemoryDataset
from src.datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
import re
import copy

#from typing import Any, Sequence
#import pickle
#from rdkit import Chem
#from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
#import numpy as np


def generate_adjacency_matrix(x, HH_COSY):
    n = x.shape[0]
    c = HH_COSY.shape[0]
    adjacency_matrix = torch.zeros((n, n), dtype=torch.int)

    for i in range(c):
        s0 = HH_COSY[i, 0]
        s1 = HH_COSY[i, 1]

        indices_s0 = torch.where(x[:, 0] == s0)[0]
        indices_s1 = torch.where(x[:, 1] == s1)[0]

        for index0 in indices_s0:
            for index1 in indices_s1:
                adjacency_matrix[index0, index1] = 6

    return adjacency_matrix




def check_mf_and_shifts_base(mf, shifts):
    shifts = list(shifts)
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", mf)
    atom_dict =  {'C', 'H', 'O', 'N'}
    atom_counts = {atom: 0 for atom in atom_dict}

    for atom, count in matches:
        if atom not in atom_dict:
            raise ValueError(f"Invalid atom {atom} in molecular formula")
        atom_counts[atom] = int(count) if count else 1
    count_H = 0
    if atom_counts['C'] != len(shifts):
        raise ValueError(f"Number of C atoms in molecular formula ({atom_counts['C']}) does not match the number of C shifts ({len(shifts)})")
    for shift_list in shifts:
        for shift in shift_list:
            if not isinstance(shift, (int, float)):
                raise ValueError(f"Invalid shift value {shift}")
    for subshift in range(len(shifts)):
        if len(shifts[subshift]) > 4:
            raise ValueError(f"Invalid shift value {shifts[subshift]}")
        if len(shifts[subshift]) > 1:
            nun_shifts = len(shifts[subshift])
            count_H += (nun_shifts-1)
    if count_H > atom_counts['H']:
        raise ValueError(f"Number of H atoms in molecular formula ({atom_counts['H']}) does not match the number of H shifts ({count_H})")

    return True

def mf_and_shifts_to_tensor_base(mf, shifts):
    atom_encoder = {'H': 0, 'C': 1, 'CH': 2, 'CH2': 3, 'CH3': 4, 'N': 5, 'O': 6}
    shifts = list(shifts)
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", mf)
    atom_dict =  {'C', 'H', 'O', 'N'}
    atom_counts = {atom: 0 for atom in atom_dict}
    for atom, count in matches:
        atom_counts[atom] = int(count) if count else 1
    num_ch= 0
    for subshift in range(len(shifts)):
        if len(shifts[subshift]) > 1:
            num_ch += (len(shifts[subshift])-1)
    print(f"Number of hydrogen atoms with chemical shifts (non-exchangeable hydrogen): {num_ch}")
    num_exchangeable_hydrogen= atom_counts['H'] - num_ch
    print(f"Number of hydrogen atoms without chemical shifts (exchangeable hydrogen): {num_exchangeable_hydrogen}")
    num_nodes = atom_counts['C'] + atom_counts['O'] + atom_counts['N'] + num_exchangeable_hydrogen
    atom_type_list = []
    atom_shifts_list = []
    for subshift in range(len(shifts)):
        if len(shifts[subshift]) == 1:
            atom_type_list.append(atom_encoder['C'])
            atom_shifts_list.append([shifts[subshift][0],0])
        elif len(shifts[subshift]) == 2:
            atom_type_list.append(atom_encoder['CH'])
            atom_shifts_list.append([shifts[subshift][0],shifts[subshift][1]])
        elif len(shifts[subshift]) == 3:
            atom_type_list.append(atom_encoder['CH2'])
            H_avg = (shifts[subshift][1] + shifts[subshift][2])/2
            atom_shifts_list.append([shifts[subshift][0],H_avg])
        elif len(shifts[subshift]) == 4:
            atom_type_list.append(atom_encoder['CH3'])
            H_avg = (shifts[subshift][1] + shifts[subshift][2] + shifts[subshift][3])/3
            atom_shifts_list.append([shifts[subshift][0],H_avg])
        else:
            raise ValueError(f"Invalid shift value {shifts[subshift]}")

    padding = num_nodes - len(atom_shifts_list)
    for i in range(padding):
        atom_shifts_list.append([0,0])
    if padding != (atom_counts['O'] + atom_counts['N'] + num_exchangeable_hydrogen):
        raise ValueError(f"The number of O, N and exchangeable hydrogen atoms is {atom_counts['O'] + atom_counts['N'] + num_exchangeable_hydrogen} but the number of padding atoms is {padding}")
    for i in range(atom_counts['O']):
        atom_type_list.append(atom_encoder['O'])
    for i in range(atom_counts['N']):
        atom_type_list.append(atom_encoder['N'])
    for i in range(num_exchangeable_hydrogen):
        atom_type_list.append(atom_encoder['H'])
    x = torch.tensor(atom_type_list).view(-1,1)
    x_label = torch.tensor(atom_shifts_list).view(-1, 2)
    x = F.one_hot(x, num_classes=len(atom_encoder)).to(torch.float).reshape(-1, len(atom_encoder))
    assert x.shape[0] == x_label.shape[0]
    return x, x_label

def check_mf_and_shifts_advanced(mf, shifts, num_OH, num_NH, num_NH2):
    shifts = list(shifts)
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", mf)
    atom_dict =  {'C', 'H', 'O', 'N'}
    atom_counts = {atom: 0 for atom in atom_dict}

    for atom, count in matches:
        if atom not in atom_dict:
            raise ValueError(f"Invalid atom {atom} in molecular formula")
        atom_counts[atom] = int(count) if count else 1
    count_H = 0
    if atom_counts['C'] != len(shifts):
        raise ValueError(f"Number of C atoms in molecular formula ({atom_counts['C']}) does not match the number of C shifts ({len(shifts)})")
    for shift_list in shifts:
        for shift in shift_list:
            if not isinstance(shift, (int, float)):
                raise ValueError(f"Invalid shift value {shift}")
    for subshift in range(len(shifts)):
        if len(shifts[subshift]) > 4:
            raise ValueError(f"Invalid shift value {shifts[subshift]}")
        if len(shifts[subshift]) > 1:
            nun_shifts = len(shifts[subshift])
            count_H += (nun_shifts-1)
    if count_H > atom_counts['H']:
        raise ValueError(f"Number of H atoms in molecular formula ({atom_counts['H']}) does not match the number of H shifts ({count_H})")
    if num_OH + num_NH + num_NH2 * 2 > atom_counts['H']:
        raise ValueError(f"Number of OH, NH and NH2 atoms ({num_OH + num_NH + num_NH2 * 2}) does not match the number of H atoms ({atom_counts['H']})")
    return True


def mf_and_shifts_to_tensor_advanced(mf, shifts, num_OH, num_NH, num_NH2):
    atom_encoder = {'C': 0, 'CH': 1, 'CH2': 2, 'CH3': 3, 'N': 4, 'NH': 5, 'NH2': 6, 'O': 7, 'OH': 8}
    shifts = list(shifts)
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", mf)
    atom_dict =  {'C', 'H', 'O', 'N'}
    atom_counts = {atom: 0 for atom in atom_dict}
    for atom, count in matches:
        atom_counts[atom] = int(count) if count else 1
    num_ch= 0
    advanced_atom_dict = {'C', 'CH', 'CH2', 'CH3', 'N', 'NH', 'NH2', 'O', 'OH'}
    advanced_atom_counts = {atom: 0 for atom in advanced_atom_dict}
    advanced_atom_counts['OH'] = num_OH
    advanced_atom_counts['NH'] = num_NH
    advanced_atom_counts['NH2'] = num_NH2
    num_O = atom_counts['O'] - num_OH
    num_N = atom_counts['N'] - num_NH - num_NH2
    advanced_atom_counts['O'] = num_O
    advanced_atom_counts['N'] = num_N
    for subshift in range(len(shifts)):
        if len(shifts[subshift]) == 1:
            advanced_atom_counts['C'] += 1
        elif len(shifts[subshift]) == 2:
            advanced_atom_counts['CH'] += 1
        elif len(shifts[subshift]) == 3:
            advanced_atom_counts['CH2'] += 1
        elif len(shifts[subshift]) == 4:
            advanced_atom_counts['CH3'] += 1
        else:
            raise ValueError(f"Invalid shift value {shifts[subshift]}")
    print(f"Number of C atoms: {advanced_atom_counts['C']}")
    print(f"Number of CH groups: {advanced_atom_counts['CH']}")
    print(f"Number of CH2 groups: {advanced_atom_counts['CH2']}")
    print(f"Number of CH3 groups: {advanced_atom_counts['CH3']}")
    print(f"Number of N atoms: {advanced_atom_counts['N']}")
    print(f"Number of NH groups: {advanced_atom_counts['NH']}")
    print(f"Number of NH2 groups: {advanced_atom_counts['NH2']}")
    print(f"Number of O atoms: {advanced_atom_counts['O']}")
    print(f"Number of OH groups: {advanced_atom_counts['OH']}")

    for subshift in range(len(shifts)):
        if len(shifts[subshift]) > 1:
            num_ch += (len(shifts[subshift])-1)
    num_nodes = (advanced_atom_counts['C'] + advanced_atom_counts['CH'] +
                 advanced_atom_counts['CH2'] + advanced_atom_counts['CH3'] +
                 advanced_atom_counts['N'] + advanced_atom_counts['NH'] +
                 advanced_atom_counts['NH2'] + advanced_atom_counts['O'] +
                 advanced_atom_counts['OH'])

    atom_type_list = []
    atom_shifts_list = []
    for subshift in range(len(shifts)):
        if len(shifts[subshift]) == 1:
            atom_type_list.append(atom_encoder['C'])
            atom_shifts_list.append([shifts[subshift][0],0])
        elif len(shifts[subshift]) == 2:
            atom_type_list.append(atom_encoder['CH'])
            atom_shifts_list.append([shifts[subshift][0],shifts[subshift][1]])
        elif len(shifts[subshift]) == 3:
            atom_type_list.append(atom_encoder['CH2'])
            H_avg = (shifts[subshift][1] + shifts[subshift][2])/2
            atom_shifts_list.append([shifts[subshift][0],H_avg])
        elif len(shifts[subshift]) == 4:
            atom_type_list.append(atom_encoder['CH3'])
            H_avg = (shifts[subshift][1] + shifts[subshift][2] + shifts[subshift][3])/3
            atom_shifts_list.append([shifts[subshift][0],H_avg])
        else:
            raise ValueError(f"Invalid shift value {shifts[subshift]}")

    padding = num_nodes - len(atom_shifts_list)
    for i in range(padding):
        atom_shifts_list.append([0,0])
    if padding != (advanced_atom_counts['O'] + advanced_atom_counts['N'] + num_OH + num_NH + num_NH2):
        raise ValueError(f"The number of O, N, NH, NH2 is {advanced_atom_counts['O'] + advanced_atom_counts['N'] + num_OH + num_NH + num_NH2} but the number of padding atoms is {padding}")
    for i in range(advanced_atom_counts['O']):
        atom_type_list.append(atom_encoder['O'])
    for i in range(advanced_atom_counts['N']):
        atom_type_list.append(atom_encoder['N'])
    for i in range(num_OH):
        atom_type_list.append(atom_encoder['OH'])
    for i in range(num_NH):
        atom_type_list.append(atom_encoder['NH'])
    for i in range(num_NH2):
        atom_type_list.append(atom_encoder['NH2'])
    x = torch.tensor(atom_type_list).view(-1,1)
    x_label = torch.tensor(atom_shifts_list).view(-1, 2)
    x = F.one_hot(x, num_classes=len(atom_encoder)).to(torch.float).reshape(-1, len(atom_encoder))
    assert x.shape[0] == x_label.shape[0]
    return x, x_label

def perturb_input_list(HSQC_list, COSY_list, C_val=5, H_val=1):
    per_HSQC = copy.deepcopy(HSQC_list)
    per_COSY = copy.deepcopy(COSY_list)
    C_shifts_list = []
    H_shifts_list = []
    for i in range(len(HSQC_list)):
        C_shift = HSQC_list[i][0]
        H_shift = HSQC_list[i][1:]
        C_shifts_list.append(C_shift)
        for j in range(len(H_shift)):
            H_shifts_list.append(H_shift[j])
    C_shifts_set = set(C_shifts_list)
    H_shifts_set = set(H_shifts_list)
    v1 = len(C_shifts_set)
    v2 = len(H_shifts_set)


    ####
    offset_c_list = torch.rand(len(C_shifts_set)) * 2 * C_val - C_val
    offset_h_list = torch.rand(len(H_shifts_set)) * 2 * H_val - H_val
    offset_c_list = offset_c_list.tolist()
    offset_h_list = offset_h_list.tolist()
    offset_c_list = [round(i, 2) for i in offset_c_list]
    offset_h_list = [round(i, 2) for i in offset_h_list]
    offset_c_dict = {}
    for i, c_shift in enumerate(C_shifts_set):
        offset_c_dict[c_shift] = offset_c_list[i]
    offset_h_dict = {}
    for i, h_shift in enumerate(H_shifts_set):
        offset_h_dict[h_shift] = offset_h_list[i]
    for i in range(len(HSQC_list)):
        C_shift = HSQC_list[i][0]
        H_shift = HSQC_list[i][1:]

        C__shift = round(C_shift + offset_c_dict[C_shift], 2)
        per_HSQC[i][0] = C__shift
        for j in range(len(H_shift)):
            H__shift = round(H_shift[j] + offset_h_dict[H_shift[j]], 2)
            per_HSQC[i][j + 1] = H__shift
    for i in range(len(COSY_list)):
        C1_shift = COSY_list[i][0]
        C2_shift = COSY_list[i][1]
        C1__shift = round(C1_shift + offset_c_dict[C1_shift], 2)
        C2__shift = round(C2_shift + offset_c_dict[C2_shift], 2)
        per_COSY[i][0] = C1__shift
        per_COSY[i][1] = C2__shift

    per_c_list = []
    per_h_list = []
    for i in range(len(per_HSQC)):
        C_shift = per_HSQC[i][0]
        H_shift = per_HSQC[i][1:]
        per_c_list.append(C_shift)
        for j in range(len(H_shift)):
            per_h_list.append(H_shift[j])
    per_c_set = set(per_c_list)
    per_h_set = set(per_h_list)
    v3 = len(per_c_set)
    v4 = len(per_h_set)
    #####
    if v1 == v3  and v2 == v4:
        return per_HSQC, per_COSY
    else:
        return False, False



class CasePredDataset(InMemoryDataset):
    def __init__(self, cfg, root, transform=None, pre_transform=None, pre_filter=None):
        """
        root: data directory
        """
        self.cfg = cfg
        # if Pretrained dir in Di-x-x, use Path(*Path(self.cfg.general.test_only).parts[:3])
        #self.pred_processed_path1 =Path(*Path(self.cfg.general.test_only).parts[:4]) / 'data' / f'{self.cfg.general.name}' / 'processed'
        self.pred_processed_path1 = Path(
            *Path(self.cfg.general.test_only).parts[:4]) / 'data' / f'{self.cfg.general.name}'

        self.pred_processed_path2= self.pred_processed_path1 / 'processed' /f'processed_{self.cfg.general.name}_pred.pt'
        #print(self.pred_processed_path1)
        #print(self.pred_processed_path2)
        #print(root)
        #assert False
        super().__init__(root=self.pred_processed_path1,transform=None, pre_transform=None, pre_filter=None)
        self.data, self.slices = torch.load(self.pred_processed_path2,weights_only=False)

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return f'processed_{self.cfg.general.name}_pred.pt'
    def process(self):
        ID=0
        RDLogger.DisableLog('rdApp.*')
        data_list = []
        infer_start_id = self.cfg.train.infer_start_id
        multi_infer = self.cfg.train.multi_infer
        print(f"The number of inferences to be run: {multi_infer}")
        if 'advanced-prediction' in self.cfg.general.name or 'augment-pred' in self.cfg.general.name:
            if '.csv' in self.cfg.prediction_input.multi_molecules_input_files :
                print("Using Batched molecules input !!!")
            else:
                #print(self.cfg)
                #print('This print is in pred_dataset.py, line 127')
                print("Using advanced-model single molecular input !!!")
                for _ in range(1):
                    molecular_formula = self.cfg.prediction_input.molecular_formula
                    shifts_values = self.cfg.prediction_input.C_H_NMR_shifts_and_HSQC
                    HH_COSY = self.cfg.prediction_input.HH_COSY
                    num_OH = self.cfg.prediction_input.num_OH
                    num_NH = self.cfg.prediction_input.num_NH
                    num_NH2 = self.cfg.prediction_input.num_NH2
                    #per_HSQC, per_COSY = perturb_input_list(shifts_values, HH_COSY)
                    per_HSQC = shifts_values
                    per_COSY = HH_COSY
                    if per_HSQC != False and per_COSY != False:
                        print("Molecular Formula: ", molecular_formula)
                        print("Shifts Values: ", shifts_values)
                        print(per_HSQC)
                        print("HH_COSY: ", HH_COSY)
                        print(per_COSY)
                        print("Number of OH: ", num_OH)
                        print("Number of NH: ", num_NH)
                        print("Number of NH2: ", num_NH2)
                        print('-'*50)
                        assert (check_mf_and_shifts_advanced(molecular_formula, shifts_values, num_OH, num_NH, num_NH2))
                        for i in range(len(HH_COSY)):
                            if len(HH_COSY[i]) != 2:
                                raise ValueError(f"Invalid HH_COSY value {HH_COSY[i]}")
                        x, x_label = mf_and_shifts_to_tensor_advanced(molecular_formula, per_HSQC, num_OH, num_NH, num_NH2)
                        y = torch.zeros((1, 0), dtype=torch.float)
                        subgraph_mask = torch.tensor(per_COSY).view(-1, 2)
                        data = Data(x=x, edge_index=None, edge_attr=None, y=y, x_label=x_label, smiles=None,
                            subgraph_mask=subgraph_mask, ID=ID,MF=molecular_formula, num_cosy=len(HH_COSY))
                        for i in tqdm(range(multi_infer)):
                            data_list.append(data)
                    else:
                        pass

        else:
            raise ValueError(f"Dataset {self.cfg.general.name} not supported")
        data, slices = self.collate(data_list)
        if not osp.exists(self.pred_processed_path1):
            os.makedirs(self.pred_processed_path1)
        torch.save((data, slices), self.pred_processed_path2)

class CasePredDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        transform = None

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train':None,
                    'val':None,
                    'test': CasePredDataset(cfg=cfg, root=root_path, transform=transform)}
        super().__init__(cfg, datasets)


class CasePredinfos(AbstractDatasetInfos):
    def __init__(self, cfg):
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        name = cfg.general.name
        if 'qm9nmr-augment' in name:
            self.name = 'qm9nmr-augment'
            self.atom_encoder = {'C': 0, 'CH': 1, 'CH2': 2, 'CH3': 3, 'N': 4, 'NH': 5, 'NH2': 6, 'O': 7, 'OH': 8}
            self.atom_decoder = ['C', 'CH', 'CH2', 'CH3', 'N', 'NH', 'NH2', 'O', 'OH']
            self.valencies = [4, 1, 2, 3, 3, 2, 1, 2, 1]
            self.edge_types = torch.tensor(cfg.dataset.edge_types, dtype=torch.float)

        elif 'pcqm4mv2nmr-advanced' in name:
            self.name = 'pcqm4mv2nmr-advanced'
            self.atom_encoder = {'C': 0, 'CH': 1, 'CH2': 2, 'CH3': 3, 'N': 4, 'NH': 5, 'NH2': 6, 'O': 7, 'OH': 8}
            self.atom_decoder = ['C', 'CH', 'CH2', 'CH3', 'N', 'NH', 'NH2', 'O', 'OH']
            self.valencies = [4, 1, 2, 3, 3, 2, 1, 2, 1]
            self.edge_types = torch.tensor(cfg.dataset.edge_types, dtype=torch.float)
        else:
            raise ValueError(f"Dataset {name} not supported")
        super().complete_infos()


