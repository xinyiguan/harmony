# Created by Xinyi Guan in 2022.
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Literal, List

import pandas as pd
from pitchtypes import SpelledPitchClass, SpelledIntervalClass


@dataclass
class Key:
    root: SpelledPitchClass
    mode: Literal['M', 'm']

    @classmethod
    def parse(cls, key_str: str) -> Key:

        _key_regex = re.compile("^(?P<class>[A-G])(?P<modifiers>(b*)|(#*))$", re.I)  # case-insensitive

        if not isinstance(key_str, str):
            raise TypeError(f"expected string as input, got {key_str}")
        key_match = _key_regex.match(key_str)
        if key_match is None:
            raise ValueError(f"could not match '{key_str}' with regex: '{_key_regex.pattern}'")
        mode = 'M' if key_match['class'].isupper() else 'm'
        root = SpelledPitchClass(key_match['class'].upper() + key_match['modifiers'])
        instance = cls(root=root, mode=mode)
        return instance

    def find_pc(self, degree) -> SpelledPitchClass:
        M_intervals = ['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7']
        m_intervals = ['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7']

        if self.mode == 'M':
            intervals = M_intervals
        elif self.mode == 'm':
            intervals = m_intervals
        else:
            raise ValueError(f'{self.mode=}')
        interval = intervals[degree - 1]
        pc = self.root + SpelledIntervalClass(interval)
        return pc

    def to_str(self) -> str:
        if self.mode == 'm':
            resulting_str = str(self.root).lower()
        elif self.mode == 'M':
            resulting_str = str(self.root)
        else:
            raise ValueError(f'not applicable mode')
        return resulting_str


@dataclass
class SingleNumeral:
    key: Key
    numeral_str: str
    degree: int
    alteration: int
    quality: Literal['M', 'm']

    _s_numeral_regex = re.compile("^(?P<modifiers>(b*)|(#*))(?P<roman_numeral>(IX|IV|V?I{0,3}))$", re.I)

    @classmethod
    def parse(cls, key_str: str | Key, numeral_str: str) -> SingleNumeral:
        numeral_scale_degree_dict = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
                                     "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}

        _s_numeral_regex = re.compile("^(?P<modifiers>(b*)|(#*))(?P<roman_numeral>(IX|IV|V?I{0,3}))$", re.I)

        if not isinstance(numeral_str, str):
            raise TypeError(f"expected string as input, got {numeral_str}")
        # match with regex
        s_numeral_match = SingleNumeral._s_numeral_regex.match(numeral_str)
        if s_numeral_match is None:
            raise ValueError(
                f"could not match '{numeral_str}' with regex: '{SingleNumeral._s_numeral_regex.pattern}'")
        if not isinstance(key_str, Key):
            key = Key.parse(key_str=key_str)
        else:
            key = key_str
        degree = numeral_scale_degree_dict.get(s_numeral_match['roman_numeral'])
        quality = "M" if s_numeral_match['roman_numeral'].isupper() else "m"

        modifiers_count = len(s_numeral_match['modifiers'])
        if all((char == "#" for char in s_numeral_match['modifiers'])):
            alteration = modifiers_count
        elif all((char == "b" for char in s_numeral_match['modifiers'])):
            alteration = -modifiers_count
        else:
            raise ValueError(f'Unexpected mixed accidentals')

        instance = cls(key=key, degree=degree, alteration=alteration, quality=quality, numeral_str=numeral_str)
        return instance

    def root(self) -> SpelledPitchClass:

        scale_pitch = self.key.find_pc(self.degree)

        if self.alteration > 0:
            alteration_symbol = 'a' * self.alteration
        elif self.alteration == 0:
            alteration_symbol = 'P'
        elif self.alteration < 0:
            alteration_symbol = 'd'
        else:
            raise ValueError(self.alteration)

        interval_class = SpelledIntervalClass(f'{alteration_symbol}1')

        root = scale_pitch + interval_class
        return root

    def key_if_tonicized(self) -> Key:
        """
        Make the current numeral as the tonic, return the spelled pitch class of the root as Key.
        """
        key = Key(root=self.root(), mode=self.quality)
        return key


@dataclass
class Numeral:
    key: Key
    L: SingleNumeral
    R: Numeral | None

    @classmethod
    def parse(cls, key_str: str, numeral_str: str) -> Numeral:
        # numeral_str examples: "#ii/V", "##III/bIV/V", "bV"
        if not isinstance(key_str, str):
            raise TypeError(f"expected string as input, got {key_str}")
        if not isinstance(numeral_str, str):
            raise TypeError(f"expected string as input, got {numeral_str}")

        key = Key.parse(key_str=key_str)
        if "/" in numeral_str:
            L_numeral_str, R_numeral_str = numeral_str.split("/", maxsplit=1)
            R = cls.parse(key_str=key_str, numeral_str=R_numeral_str)
            L = SingleNumeral.parse(key_str=R.key_if_tonicized(), numeral_str=L_numeral_str)

        else:
            L = SingleNumeral.parse(key_str=key_str, numeral_str=numeral_str)
            R = None

        instance = cls(key=key, L=L, R=R)
        return instance

    def quality(self):
        return self.L.quality

    def parent_key(self):
        if self.R is None:
            return self.key
        else:
            return self.R.key_if_tonicized()

    def root(self):
        return self.L.root()

    def key_if_tonicized(self) -> Key:
        key_if_tonicized = Key(root=self.root(), mode=self.quality())
        return key_if_tonicized

    def applied_chord_to_roman_numeral(self) -> SingleNumeral:
        """
        Applied chords are converted into Roman numerals relative to the key, e.g. V/V becomes II
        """
        raise NotImplementedError


@dataclass
class ModulationBigram:
    #  The constructor takes a string consisting of the form
    #  <key><source_numeral><target_numeral>, e.g. "C_bIII/V_ii/bIII/V"
    key: Key
    source: Numeral
    target: Numeral

    @classmethod
    def parse(cls, modulation_bigram_str: str) -> ModulationBigram:
        # example of modulation_bigram_str: 'C_bIII/V_ii/bIII/V'
        _modulation_bigram_regex = re.compile('.*_.*_.*')
        if not isinstance(modulation_bigram_str, str):
            raise TypeError(f"expected string as input, got {modulation_bigram_str}")

        # match with regex
        modulation_bigram_match = _modulation_bigram_regex.match(modulation_bigram_str)
        if modulation_bigram_match is None:
            raise ValueError(
                f"could not match '{modulation_bigram_str}' with regex: '{_modulation_bigram_regex.pattern}'")

        key_str, source_str, target_str = modulation_bigram_str.split("_")

        key = Key.parse(key_str=key_str)
        source = Numeral.parse(key_str=key_str, numeral_str=source_str)
        target = Numeral.parse(key_str=key_str, numeral_str=target_str)

        instance = cls(key=key, source=source, target=target)
        return instance

    def type(self):
        source_quality = self.source.quality()
        target_quality = self.target.quality()
        modulation_type = f'{source_quality}{target_quality}'
        return modulation_type

    def interval(self) -> SpelledIntervalClass:
        source_root = self.source.root()
        target_root = self.target.root()
        modulation_step = target_root - source_root
        return modulation_step


@dataclass
class Progression:
    """A class of harmonic progression (Roman Numerals) occurs within a local key region"""
    globalkey: Key
    preceding_key: Key | None
    following_key: Key | None

    localkey: SingleNumeral
    roots: List[str]
    roman_numerals: List[str]
    chords: List[str]

    @classmethod
    def parse(cls, key_region_df: pd.DataFrame) -> Progression:
        """
        Takes the key_region_subdf from the PieceInfo.
        columns: ["globalkey", "localkey", "chord", "numeral", "form", "figbass", "changes", "relativeroot",
                    "root", "bass_note", "key_region_label"]
        """
        if not isinstance(key_region_df, pd.DataFrame):
            raise TypeError(f"expected pd.DataFrame as input, got {key_region_df}")

        globalkey_str = key_region_df['globalkey'].tolist()[0]
        globalkey = Key.parse(key_str=globalkey_str)
        localkey = SingleNumeral.parse(key_str=globalkey, numeral_str=key_region_df['localkey'].tolist()[0])
        root = key_region_df['root'].tolist()
        numeral = key_region_df['numeral'].tolist()
        chords = key_region_df['chord'].tolist()

        instance = cls(globalkey=globalkey, localkey=localkey, roots=root, roman_numerals=numeral, chords=chords)
        return instance

    def length(self) -> int:
        return len(self.roots)


if __name__ == '__main__':
    single_numeral = SingleNumeral.parse(key_str="c", numeral_str="bIII")
    print(single_numeral.key_if_tonicized())
    result = single_numeral.key_if_tonicized().to_str()

    print(result)
