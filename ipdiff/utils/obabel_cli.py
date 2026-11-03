# Copyright (c) 2026 CREF contributors
# SPDX-License-Identifier: MIT
"""Use separately installed chemistry tools through standard file formats."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem


def convert(text, input_format, output_format, options=()):
    """Run obabel as a separate process, using stdin/stdout instead of bindings."""
    program = shutil.which(os.environ.get('CREF_OBABEL', 'obabel'))
    if program is None:
        raise RuntimeError('obabel is not on PATH; install Open Babel or set CREF_OBABEL.')
    try:
        result = subprocess.run(
            [program, '-i' + input_format, '-o' + output_format] + list(options),
            input=text, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace', timeout=30, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('obabel conversion exceeded 30 seconds.') from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError('obabel conversion failed: ' + result.stderr.strip()[:800])
    return result.stdout


def receptor_preparer():
    """Locate the installed receptor-preparation script without importing it."""
    override = os.environ.get('CREF_PREPARE_RECEPTOR')
    if override:
        path = Path(override)
    else:
        spec = importlib.util.find_spec('AutoDockTools')
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            raise RuntimeError('Install AutoDockTools or set CREF_PREPARE_RECEPTOR.')
        path = Path(next(iter(locations))) / 'Utilities24' / 'prepare_receptor4.py'
    if not path.is_file():
        raise RuntimeError('Receptor preparation script not found: ' + str(path))
    return str(path)


class LigandPreparation:
    """Prepare a ligand through RDKit and the external obabel PDBQT writer."""

    def __init__(self, input_mol, mol_format):
        if mol_format == 'sdf':
            self.molecule = next(iter(Chem.SDMolSupplier(str(input_mol), removeHs=False)), None)
        elif mol_format == 'smi':
            self.molecule = Chem.MolFromSmiles(input_mol)
        else:
            raise ValueError('Expected an SDF file or SMILES string.')
        if self.molecule is None or self.molecule.GetNumAtoms() == 0:
            raise ValueError('Ligand could not be parsed.')
        if len(Chem.GetMolFrags(self.molecule)) != 1:
            raise ValueError('Vina ligand preparation requires one connected component.')

    def addH(self, polaronly=False, correctforph=True, PH=7):
        options = ['--addpolarh' if polaronly else '-h']
        if correctforph:
            options += ['-p', str(float(PH))]
        text = convert(Chem.MolToMolBlock(self.molecule) + '\n$$$$\n', 'sdf', 'sdf', options)
        molecule = Chem.MolFromMolBlock(text.split('$$$$', 1)[0], removeHs=False)
        if molecule is None:
            raise ValueError('Hydrogen preparation produced an invalid molecule.')
        self.molecule = molecule

    def gen_conf(self):
        self.molecule = Chem.AddHs(self.molecule)
        if AllChem.EmbedMolecule(self.molecule, AllChem.ETKDGv3()) != 0:
            raise ValueError('Ligand conformer generation failed.')

    def get_pdbqt(self, lig_pdbqt=None):
        if self.molecule.GetNumConformers() == 0:
            self.gen_conf()
        molecule = Chem.AddHs(self.molecule, addCoords=True)
        text = convert(Chem.MolToMolBlock(molecule) + '\n$$$$\n', 'sdf', 'pdbqt',
                       ['--partialcharge', 'gasteiger'])
        if not any(line.startswith(('ATOM  ', 'HETATM')) for line in text.splitlines()):
            raise ValueError('Ligand preparation produced an empty PDBQT file.')
        tags = [line.strip() for line in text.splitlines()]
        if tags.count('ROOT') != 1 or tags.count('ENDROOT') != 1:
            raise ValueError('Ligand PDBQT must contain exactly one torsion tree.')
        if lig_pdbqt is None:
            return text
        Path(lig_pdbqt).write_text(text, encoding='utf-8')
