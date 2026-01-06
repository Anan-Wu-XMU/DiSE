import torch
from src import utils


class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        self.charge = ChargeFeature(valencies=dataset_infos.valencies,name = dataset_infos.name)
        self.valency = ValencyFeature(name=dataset_infos.name)

    def __call__(self, noisy_data):
        charge = self.charge(noisy_data).unsqueeze(-1)      # (bs, n, 1)
        valency = self.valency(noisy_data).unsqueeze(-1)    # (bs, n, 1)

        extra_edge_attr = torch.zeros((*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t'])
        y = noisy_data['y_t']
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.PlaceHolder(X=torch.cat((charge, valency), dim=-1), E=extra_edge_attr, y=empty_y)


class ChargeFeature:
    def __init__(self, valencies, name):
        self.valencies = valencies
        self.name = name

    def __call__(self, noisy_data):
        if 'qm9nmr-augment' in self.name:
            bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        elif 'pcqm4mv2nmr-advanced' in self.name:
            bond_orders = torch.tensor([0, 1, 2, 3, 1.5, 1], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        else:
            raise NotImplementedError("Unknown dataset {}".format(self.name))
        weighted_E = noisy_data['E_t'] * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(self.valencies, device=noisy_data['X_t'].device).reshape(1, 1, -1)
        X = noisy_data['X_t'] * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)               # (bs, n)

        return (normal_valencies - current_valencies).type_as(noisy_data['X_t'])


class ValencyFeature:
    def __init__(self, name):
        self.name = name

    def __call__(self, noisy_data):
        if 'qm9nmr-augment' in self.name:
            orders = torch.tensor([0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        elif 'pcqm4mv2nmr-advanced' in self.name:
            orders = torch.tensor([0, 1, 2, 3, 1.5, 1], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        else:
            raise NotImplementedError("Unknown dataset {}".format(self.name))
        E = noisy_data['E_t'] * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.type_as(noisy_data['X_t'])