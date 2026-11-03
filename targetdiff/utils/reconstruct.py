# Copyright (c) 2026 CREF contributors
# SPDX-License-Identifier: MIT
"""CREF reconstruction adapter using standard formats and public RDKit APIs.

Coordinate-only inputs use the separately installed obabel executable for bond
perception. Explicit bond graphs are constructed and sanitized with RDKit.
This replaces the former inherited reconstruction source, not its heuristics.
"""

import numpy as np
from rdkit import Chem

from utils.obabel_cli import convert


BACKEND = 'cref-standard-formats-v1'


class MolReconsError(ValueError):
    """The supplied atoms could not be reconstructed as a valid molecule."""


def _inputs(xyz, atomic_nums):
    try:
        coordinates = np.asarray(xyz, dtype=float)
        numbers = np.asarray(atomic_nums)
        if numbers.ndim != 1 or len(numbers) == 0:
            raise ValueError('Expected a nonempty atomic-number vector.')
        integers = numbers.astype(int)
        if not np.array_equal(numbers, integers) or np.any((integers < 1) | (integers > 118)):
            raise ValueError('Atomic numbers must be integers between 1 and 118.')
        if coordinates.shape != (len(integers), 3) or not np.isfinite(coordinates).all():
            raise ValueError('Expected finite N-by-3 coordinates.')
        return coordinates, integers.tolist()
    except (TypeError, ValueError, OverflowError) as exc:
        raise MolReconsError(str(exc)) from exc


def _restore_positions(molecule, coordinates, numbers):
    if [atom.GetAtomicNum() for atom in molecule.GetAtoms()] != numbers:
        raise MolReconsError('Reconstruction changed the input atom count or ordering.')
    conformer = Chem.Conformer(len(numbers))
    for index, point in enumerate(coordinates):
        conformer.SetAtomPosition(index, tuple(float(value) for value in point))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    molecule.SetProp('_CREF_reconstruction_backend', BACKEND)
    return molecule


def reconstruct_from_generated(xyz, atomic_nums, aromatic=None, basic_mode=True):
    """Perceive bonds via XYZ -> SDF; supplied coordinates and order are retained.

    Aromaticity is perceived by the chemistry tools. The aromatic/basic_mode
    arguments remain accepted for caller compatibility; aromatic hints are not
    forced into the resulting graph. This is not the legacy reconstruction.
    """
    coordinates, numbers = _inputs(xyz, atomic_nums)
    if aromatic is not None and np.asarray(aromatic).shape != (len(numbers),):
        raise MolReconsError('Aromatic hints must have one entry per atom.')
    table = Chem.GetPeriodicTable()
    rows = [str(len(numbers)), 'CREF coordinate reconstruction']
    rows.extend('{} {:.12f} {:.12f} {:.12f}'.format(table.GetElementSymbol(number), *point)
                for number, point in zip(numbers, coordinates))
    try:
        text = convert('\n'.join(rows) + '\n', 'xyz', 'sdf')
        molecule = Chem.MolFromMolBlock(text.split('$$$$', 1)[0], sanitize=False, removeHs=False)
        if molecule is None:
            raise ValueError('RDKit rejected the perceived bond graph.')
        # Generated atom types describe closed-shell heavy atoms, not radicals.
        # XYZ omits their hydrogens; let RDKit assign the implicit H counts.
        for atom in molecule.GetAtoms():
            atom.SetNoImplicit(False)
            atom.SetNumRadicalElectrons(0)
            if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
                if all(bond.GetBondType() == Chem.BondType.SINGLE for bond in atom.GetBonds()):
                    atom.SetFormalCharge(1)
        Chem.SanitizeMol(molecule)
        return _restore_positions(molecule, coordinates, numbers)
    except MolReconsError:
        raise
    except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
        raise MolReconsError('Coordinate reconstruction failed: ' + str(exc)) from exc


def reconstruct_from_generated_with_bond(xyz, atomic_nums, bond_index, bond_type, check_validity=True):
    """Build the supplied graph and use standard-format normalization if needed.

    For a directed pair list, the i<j entry defines the undirected bond. A
    reverse-only entry is accepted too. Bond codes 0..4 mean absent, single,
    double, triple, and aromatic. Conflicting repeated directed entries fail.
    If RDKit rejects aromatic/valence assignments, an SDF round trip through
    the external obabel program may normalize bond orders. Atom identities,
    adjacency, and coordinates must remain unchanged; no legacy repair code
    is used. check_validity=False skips normalization for geometric guidance.
    """
    coordinates, numbers = _inputs(xyz, atomic_nums)
    try:
        edges = np.asarray(bond_index)
        labels = np.asarray(bond_type)
        if edges.shape != (2, len(labels)) or labels.ndim != 1:
            raise ValueError('Expected a 2-by-E edge array and E bond labels.')
        if not np.array_equal(edges, edges.astype(int)) or not np.array_equal(labels, labels.astype(int)):
            raise ValueError('Edge indices and bond labels must be integers.')
        directed = {}
        for pair, label in zip(edges.T.astype(int), labels.astype(int)):
            start, end = map(int, pair)
            if not (0 <= start < len(numbers) and 0 <= end < len(numbers)) or start == end:
                raise ValueError('Bond endpoints must be distinct valid atom indices.')
            if not 0 <= label <= 4:
                raise ValueError('Unsupported bond label.')
            if (start, end) in directed and directed[start, end] != label:
                raise ValueError('Conflicting duplicate bond labels.')
            directed[start, end] = int(label)
        builder = Chem.RWMol()
        for number in numbers:
            builder.AddAtom(Chem.Atom(number))
        types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
        pairs = {tuple(sorted(pair)) for pair in directed}
        for start, end in sorted(pairs):
            label = directed.get((start, end), directed.get((end, start)))
            if label:
                builder.AddBond(start, end, types[label])
                if label == 4:
                    builder.GetAtomWithIdx(start).SetIsAromatic(True)
                    builder.GetAtomWithIdx(end).SetIsAromatic(True)
        molecule = builder.GetMol()
        if check_validity:
            # A four-coordinate nitrogen without charge is conventionally N+.
            for atom in molecule.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
                    if all(bond.GetBondType() == Chem.BondType.SINGLE for bond in atom.GetBonds()):
                        atom.SetFormalCharge(1)
            original = Chem.Mol(molecule)
            try:
                Chem.SanitizeMol(molecule)
            except (RuntimeError, ValueError):
                original.UpdatePropertyCache(strict=False)
                _restore_positions(original, coordinates, numbers)
                text = Chem.MolToMolBlock(original, kekulize=False) + '\n$$$$\n'
                normalized = convert(text, 'sdf', 'sdf')
                molecule = Chem.MolFromMolBlock(normalized.split('$$$$', 1)[0], removeHs=False)
                if molecule is None:
                    raise ValueError('Standard-format bond normalization failed.')
                original_edges = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                                  for bond in original.GetBonds()}
                normalized_edges = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                                    for bond in molecule.GetBonds()}
                if original_edges != normalized_edges:
                    raise ValueError('Bond normalization changed the predicted adjacency.')
                molecule.SetBoolProp('_CREF_bond_orders_normalized', True)
        else:
            molecule.UpdatePropertyCache(strict=False)
        return _restore_positions(molecule, coordinates, numbers)
    except MolReconsError:
        raise
    except (RuntimeError, ValueError, TypeError, OverflowError) as exc:
        raise MolReconsError('Explicit-bond reconstruction failed: ' + str(exc)) from exc
