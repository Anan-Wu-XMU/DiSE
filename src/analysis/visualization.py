import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import rdkit.Chem
import wandb


class MolecularVisualization:
    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder
        mode = len(atom_decoder)
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        add_hs_idx = {}

        # Define the atom addition logic based on mode
        def add_atom(atom_type, hs_count):
            atom = Chem.Atom(atom_type)
            molIdx = mol.AddAtom(atom)
            return molIdx, hs_count

        mode_map = {4: {},
            # {'H': 0, 'C': 1, 'N': 2, 'O': 3}
            7: {2: ('C', 1), 3: ('C', 2), 4: ('C', 3)},
            # {'H': 0, 'C': 1, 'CH': 2, 'CH2': 3, 'CH3': 4, 'N': 5, 'O': 6}
            9: {1: ('C', 1), 2: ('C', 2), 3: ('C', 3), 5: ('N', 1), 6: ('N', 2), 8: ('O', 1)}
            # {'C': 0, 'CH': 1, 'CH2': 2, 'CH3': 3, 'N': 4, 'NH': 5, 'NH2': 6, 'O': 7, 'OH': 8}
        }

        for i, node in enumerate(node_list):
            if node == -1:
                continue
            if node in mode_map[mode]:
                atom_type, hs_count = mode_map[mode][node]
                molIdx, add_hs_count = add_atom(atom_type, hs_count)
                node_to_idx[i], add_hs_idx[i] = molIdx, add_hs_count
            else:
                molIdx = mol.AddAtom(Chem.Atom(atom_decoder[int(node)]))
                node_to_idx[i] = molIdx

        # Add bonds based on adjacency matrix
        bond_types = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE,
                      3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC,
                      5: Chem.rdchem.BondType.AROMATIC}

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                if iy <= ix or bond not in bond_types:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_types[bond])

        # Add hydrogen atoms
        for i, hs_count in add_hs_idx.items():
            for _ in range(hs_count):
                mol.AddBond(node_to_idx[i], mol.AddAtom(Chem.Atom('H')), Chem.rdchem.BondType.SINGLE)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None

        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
                if wandb.run and log is not None:
                    print(f"Saving {file_path} to wandb")
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")

    def visualize_true_molecules(self, path, true_molecules, num_molecules_to_visualize, log='true_graph'):
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Visualizing {num_molecules_to_visualize} of {len(true_molecules)} true molecules")
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}_true.png'.format(i))
            true_mol = true_molecules[i]
            try:
                Draw.MolToFile(true_mol, file_path)
                if wandb.run and log is not None:
                    print(f"Saving {file_path} to wandb")
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")

    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        #RDLogger.DisableLog('rdApp.*')
        # convert graphs to the rdkit molecules
        mols = [self.mol_from_graphs(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)
        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))
        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            Draw.MolToFile(mols[frame], file_name, size=(400, 400), legend=f"Frame {frame}")
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)

        if wandb.run:
            print(f"Saving {gif_path} to wandb")
            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)

        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(os.path.join(path, '{}-grid-image.png'.format(path.split('/')[-1])))
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
        return mols
