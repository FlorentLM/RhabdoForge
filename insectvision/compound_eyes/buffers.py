"""
Packed data container for compound-eye buffers.

EyesBuffer is a Data Transfer Object (DTO) holding the four packed structured arrays
that are passed to the GPU, plus the cartridge mapping.

It only knows how to allocate itself, serialize to / from disk, and bitpack the
receptor metadata field.
"""
from contextlib import contextmanager
from typing import Optional

import numpy as np

from insectvision.compound_eyes.datatypes import (
    LENS_STATIC_DTYPE,
    LENS_DYNAMIC_DTYPE,
    RCPT_STATIC_DTYPE,
    RCPT_DYNAMIC_DTYPE,
    get_metadata_field,
    set_metadata_field,
    pack_metadata,
)


class EyesBuffer:
    """
    GPU-ready packed buffers for a compound eye.

    Four numpy structured arrays:
        - lens_static_data:  (N,)   per-lens static fields
        - lens_dynamic_data: (N,)   per-lens dynamic state
        - rcpt_static_data:  (N*R,) per-receptor static fields
        - rcpt_dynamic_data: (N*R,) per-receptor dynamic state

    Plus an (N, R) cartridge mapping table and a 'lens_dirty' flag
    that downstream GPU upload code can read/clear.
    """

    def __init__(self, n_lenses: int, receptors_per_lens: int):

        if n_lenses < 1:
            raise ValueError(f"n_lenses must be >= 1, got {n_lenses}")
        if receptors_per_lens < 1:
            raise ValueError(f"receptors_per_lens must be >= 1, got {receptors_per_lens}")

        self._n_lenses = int(n_lenses)
        self._receptors_per_lens = int(receptors_per_lens)
        total_rcpt = n_lenses * receptors_per_lens

        self.lens_static_data = np.zeros(n_lenses, dtype=LENS_STATIC_DTYPE)
        self.lens_dynamic_data = np.zeros(n_lenses, dtype=LENS_DYNAMIC_DTYPE)
        self.rcpt_static_data = np.zeros(n_lenses * receptors_per_lens, dtype=RCPT_STATIC_DTYPE)
        self.rcpt_dynamic_data = np.zeros(n_lenses * receptors_per_lens, dtype=RCPT_DYNAMIC_DTYPE)

        # Cartridge mapping: (N, R)
        # Entry (i, r) is the global lens index whose r-th rhabdomere feeds the cartridge centered at lens i.
        # Identity by default
        self.cartridge_map = np.tile(
            np.arange(n_lenses, dtype=np.intp)[:, None],
            (1, receptors_per_lens),
        )
        self.cartridges_wired = False

        # GPU upload tracking: whenever lens-level data is mutated it needs to be reuploaded.
        # The renderer is expected to read and clear the flag
        self.lens_dirty = True
        self.rcpt_dirty = True
        self.lens_dirty_mask = np.ones(self._n_lenses, dtype=bool)
        self.rcpt_dirty_mask = np.ones(total_rcpt, dtype=bool)

        self._allow_lens_writes = False
        self._allow_rcpt_writes = False

    @contextmanager
    def unlock(self, lenses: Optional[bool] = None, receptors: Optional[bool] = None):
        """
        Context manager to temporarily allow CPU-side modifications to the buffers.

        If neither is specified, both are unlocked.
        If one is specified, the other remains locked.
        """

        if lenses is None and receptors is None:
            lenses = True
            receptors = True
        else:
            lenses = lenses or False
            receptors = receptors or False

        prev_lens = self._allow_lens_writes
        prev_rcpt = self._allow_rcpt_writes

        self._allow_lens_writes = prev_lens or lenses
        self._allow_rcpt_writes = prev_rcpt or receptors

        try:
            yield
        finally:
            self._allow_lens_writes = prev_lens
            self._allow_rcpt_writes = prev_rcpt

    # Sizes

    @property
    def n_lenses(self) -> int:
        return self._n_lenses

    @property
    def receptors_per_lens(self) -> int:
        return self._receptors_per_lens

    @property
    def total_receptors(self) -> int:
        return self._n_lenses * self._receptors_per_lens

    def __len__(self) -> int:
        return self._n_lenses

    def __repr__(self) -> str:
        return (f"EyesBuffer(N={self._n_lenses}, R={self._receptors_per_lens}, "
                f"cartridges_wired={self.cartridges_wired})")

    # Bitpacking helpers

    def get_metadata(self, field: str) -> np.ndarray:
        """
        Extract one bitfield from rcpt_static_data['metadata'].

        See datatypes._BIT_LAYOUT for available fields.
        """
        return get_metadata_field(self.rcpt_static_data['metadata'], field)

    def set_metadata(self, field: str, value) -> None:
        """
        In-place update of one bitfield in rcpt_static_data['metadata'].
        """
        if not self._allow_rcpt_writes:
            raise RuntimeError("CPU-side receptor writes are locked.")

        self.rcpt_static_data['metadata'] = set_metadata_field(
            self.rcpt_static_data['metadata'], field, value
        )
        self.rcpt_dirty = True
        self.rcpt_dirty_mask.fill(True)

    def pack_metadata(self,
                      eye_indices,
                      receptor_types,
                      neighbour_counts,
                      lens_indices,
                      chirality_neg,
                      ) -> None:
        """
        Replace the entire rcpt_static_data['metadata'] with packed fields.
        """
        if not self._allow_rcpt_writes:
            raise RuntimeError("CPU-side receptor writes are locked.")

        self.rcpt_static_data['metadata'] = pack_metadata(
            eye_indices=eye_indices,
            receptor_types=receptor_types,
            neighbour_counts=neighbour_counts,
            lens_indices=lens_indices,
            chirality_neg=chirality_neg,
        )
        self.rcpt_dirty = True
        self.rcpt_dirty_mask.fill(True)

    # Cartridge-index view

    @property
    def cartridge_indices(self) -> np.ndarray:
        """
        (N, R) global receptor indices grouped by cartridge mapping.

        Raises RuntimeError if cartridges have not been wired.
        """
        if not self.cartridges_wired:
            raise RuntimeError("Cartridges not wired")
        return (self.cartridge_map * self._receptors_per_lens
                + np.arange(self._receptors_per_lens, dtype=np.intp))

    # I/O

    def to_file(self, path: str) -> None:
        """
        Serialize the entire buffer state to a .npz archive.

        Writes the four structured arrays, the cartridge map, and the
        cartridges_wired flag.
        """
        np.savez(
            path,
            lens_static_data=self.lens_static_data,
            lens_dynamic_data=self.lens_dynamic_data,
            rcpt_static_data=self.rcpt_static_data,
            rcpt_dynamic_data=self.rcpt_dynamic_data,
            cartridge_map=self.cartridge_map,
            cartridges_wired=np.bool_(self.cartridges_wired),
        )

    @classmethod
    def from_file(cls, path: str) -> 'EyesBuffer':
        """
        Restore a buffer from a .npz archive previously written by to_file().

        Raises ValueError if the structured-array dtypes in the archive
        do not match the current LENS_*_DTYPE / RCPT_*_DTYPE.
        """
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            required = {'lens_static_data', 'lens_dynamic_data',
                        'rcpt_static_data', 'rcpt_dynamic_data',
                        'cartridge_map'}
            missing = required - files
            if missing:
                raise ValueError(f"{path}: missing required keys: {sorted(missing)}")

            ls = data['lens_static_data']
            ld = data['lens_dynamic_data']
            rs = data['rcpt_static_data']
            rd = data['rcpt_dynamic_data']
            cm = data['cartridge_map']

            if ls.dtype != LENS_STATIC_DTYPE:
                raise ValueError(f"{path}: lens_static_data dtype mismatch")
            if ld.dtype != LENS_DYNAMIC_DTYPE:
                raise ValueError(f"{path}: lens_dynamic_data dtype mismatch")
            if rs.dtype != RCPT_STATIC_DTYPE:
                raise ValueError(f"{path}: rcpt_static_data dtype mismatch")
            if rd.dtype != RCPT_DYNAMIC_DTYPE:
                raise ValueError(f"{path}: rcpt_dynamic_data dtype mismatch")

            n_lenses = int(ls.shape[0])
            n_receptors = int(rs.shape[0])
            if n_receptors % n_lenses != 0:
                raise ValueError(
                    f"{path}: receptor count {n_receptors} not divisible by lens count {n_lenses}"
                )
            R = n_receptors // n_lenses

            buf = cls(n_lenses=n_lenses, receptors_per_lens=R)
            buf.lens_static_data[:] = ls
            buf.lens_dynamic_data[:] = ld
            buf.rcpt_static_data[:] = rs
            buf.rcpt_dynamic_data[:] = rd
            buf.cartridge_map = np.asarray(cm, dtype=np.intp).reshape(n_lenses, R)
            buf.cartridges_wired = bool(data['cartridges_wired']) if 'cartridges_wired' in files else False
            buf.lens_dirty = True

        return buf
