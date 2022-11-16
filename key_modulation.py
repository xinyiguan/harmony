# Created by Xinyi Guan in 2022.
from __future__ import annotations
from dataclasses import dataclass
from pitchtypes import SpelledPitch, SpelledInterval


@dataclass
class Key:
    root: SpelledPitch
    quality: str

    M_intervals = ['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7']
    m_intervals = ['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7']

    def find_pc(self, degree) -> SpelledPitch:
        if self.quality == 'M':
            intervals = self.M_intervals
        elif self.quality == 'm':
            intervals = self.m_intervals
        else:
            raise ValueError
        interval = intervals[degree - 1]
        pc = self.root + SpelledInterval(interval)
        return pc


@dataclass
class Numeral:
    key: Key
    L: SingleNumeral
    R: Numeral | None

    def root(self):
        if self.parent_key is not None:
            key = self.parent_key
        else:
            key = self.key
        root = key.find_pc(degree=self.L.degree)
        return root

    def quality(self):
        return self.L.quality

    def parent_key(self):
        return self.R.outer_key()

    def outer_key(self):
        if self.R is None:
            outer_key = self.key
        else:
            outer_key = Key(root=self.root(), quality=self.quality())
        return outer_key


@dataclass
class SingleNumeral(Numeral):
    degree: int
    accidentals: int
    quality: str

    def __init__(self):
        self.L = self
        self.R = None


@dataclass
class ModulationBigram:
    # 'C_bIII/V_ii/bIII/V'
    key: Key
    source: Numeral
    target: Numeral
