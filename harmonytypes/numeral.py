from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Self, Optional, Literal, Generic, TypeVar, Dict, Tuple, List

import pandas as pd
from pitchtypes import SpelledPitchClass, SpelledPitchClassArray, aspc, SpelledIntervalClass, asic

from harmonytypes import theory
from harmonytypes.degree import Degree
from harmonytypes.key import Key
from harmonytypes.quality import TertianHarmonyQuality

T = TypeVar('T')


@dataclass
class AbstractNumeral(abc.ABC):

    @staticmethod
    def parse_numeral(s: str) -> Dict:
        """
        Parse a string as a numeral. Return a dictionary with keys: singlenumeral, type, figbass, changes, and relativeroot
        """
        if not isinstance(s, str):
            raise TypeError(f"expected string as input, got {s}")

        # match with regex
        numeral_match = theory.DCML_numeral_regex.match(s)
        if numeral_match is None:
            raise ValueError(f"could not match '{s}' with regex: '{theory.DCML_numeral_regex.pattern}'")

        result_dict = {"numeral": numeral_match["numeral"],
                       "form": numeral_match["form"],
                       "figbass": numeral_match["figbass"],
                       "changes": numeral_match["changes"],
                       "relative_root": numeral_match["relativeroot"]}
        return result_dict


@dataclass
class SimpleNumeral:
    numeral_str: str
    key: Key
    degree: Degree
    quality: Literal["major", "minor"]

    @classmethod
    def from_string(cls, numeral_str: str, key: Key) -> Self:
        degree = Degree.from_string(degree_str=numeral_str)
        quality = "major" if numeral_str.isupper() else "minor"
        instance = cls(numeral_str=numeral_str,
                       key=key,
                       degree=degree,
                       quality=quality)
        return instance

    def root(self) -> SpelledPitchClass:
        root = self.key.find_spc(degree=self.degree, key=self.key)
        return root

    def key_if_tonicized(self) -> Key:
        mode = self.quality
        if mode == "major":
            mode = mode
        elif mode == "minor":
            mode = "natural_minor"
        else:
            raise ValueError
        key = Key(tonic=self.root(), mode=mode)
        return key


@dataclass
class Chain(Generic[T]):
    head: T
    tail: Optional[Chain[T]]


@dataclass
class NumeralChain(Chain[SimpleNumeral]):
    key: Key

    @classmethod
    def parse(cls, numeral_str: str, key: Key) -> Self:

        if "/" in numeral_str:
            L_numeral_str, R_numeral_str = numeral_str.split("/", maxsplit=1)
            tail = cls.parse(key=key, numeral_str=R_numeral_str)
            head = SimpleNumeral.from_string(key=tail.head.key_if_tonicized(), numeral_str=L_numeral_str)

        else:
            head = SimpleNumeral.from_string(key=key, numeral_str=numeral_str)
            tail = None
        instance = cls(head=head, tail=tail, key=key)

        return instance

    def key_if_tonicized(self) -> Key:
        result_key = self.head.key_if_tonicized()
        return result_key

    def degree(self) -> Degree:
        """
        Return the degree of the numeral in the key.
        Example: return 6 for "ii/V" in C major.
        """
        result_key = self.key_if_tonicized()
        result_key_tonic = result_key.tonic
        degree = self.key.find_degree(result_key_tonic)
        return degree


@dataclass
class Numeral(AbstractNumeral):
    numeral_string: str
    global_key: Key
    local_key: Key
    # degree: Degree
    harmony_quality: TertianHarmonyQuality
    spcs: SpelledPitchClassArray
    root: SpelledPitchClass
    bass: SpelledPitchClass

    @classmethod
    def from_string(cls, numeral_str: str, global_key: Key, local_key: Key) -> Self:
        raise NotImplementedError

    @classmethod
    def from_df(cls, df_row: pd.DataFrame) -> Self:  # TODO: VERY MESSY!
        """Read all relevant info from the dcml tsv table"""

        def standardize_input(my_input: str | Tuple[int] | float) -> List[int]:
            if isinstance(my_input, float) and math.isnan(my_input):
                assert math.isnan(my_input)
                result = []
            elif isinstance(my_input, float):
                result = [int(my_input)]
            elif isinstance(my_input, tuple):
                result = list(my_input)
            elif isinstance(my_input, str):
                result = list(int(x) for x in my_input.split(','))
            else:
                raise TypeError(f'{my_input=} {type(my_input)=}')
            return result

        global_key: Key = Key.from_string(df_row['globalkey'])
        local_key_degree = NumeralChain.parse(numeral_str=str(df_row['localkey']), key=global_key).degree()

        local_key = Key(tonic=global_key.find_spc(degree=local_key_degree, key=global_key),
                        mode='major' if str(df_row['localkey']).isupper() else 'natural_minor')

        chord_str = str(df_row["chord"])

        # degree = Degree.from_string(degree_str=AbstractNumeral.parse_numeral(df_row["chord"])["numeral"])

        harmony_quality = TertianHarmonyQuality.from_chord_type_str(chord_type=df_row["chord_type"])
        pcs_sd_int_in_fifths_in_localkey = [int(x) for x in list(standardize_input(df_row['chord_tones']) +
                                                                 standardize_input(df_row['added_tones']))]
        pcs_in_fifths = [Key.find_spc(key=local_key, degree=Degree.from_fifth(x, mode=local_key.mode)) for x in
                         pcs_sd_int_in_fifths_in_localkey]

        pcs = aspc(things=pcs_in_fifths)

        root = Key.find_spc(key=local_key, degree=Degree.from_fifth(int(df_row['root']), mode=local_key.mode))
        bass = Key.find_spc(key=local_key, degree=Degree.from_fifth(int(df_row['bass_note']), mode=local_key.mode))

        instance = cls(numeral_string=chord_str,
                       local_key=local_key,
                       global_key=global_key,
                       harmony_quality=harmony_quality,
                       spcs=pcs,
                       root=root,
                       bass=bass)
        return instance

    def to_spelled_chord(self, reference_key: Key) -> SpelledPitchClass:
        raise NotImplementedError

    def non_diatonic_spcs(self, reference_key: Key) -> List[SpelledPitchClass]:
        ndpcs = [x for x in self.spcs if x not in reference_key.get_scale()]
        return ndpcs

    def key_if_tonicized(self) -> Key:
        mode = self.harmony_quality.major_minor_mode()
        key = Key(tonic=self.root, mode=mode)
        return key

    def third(self) -> SpelledPitchClass:
        scale_mode = self.harmony_quality.major_minor_mode
        if scale_mode == 'major':
            ic_to_add = SpelledIntervalClass('M3')
        else:
            ic_to_add = SpelledIntervalClass('m3')
        result = self.root + ic_to_add
        return result

    def fifth(self) -> SpelledPitchClass:
        the_root = self.root
        result = the_root + SpelledIntervalClass('P5')
        return result

    def diatonic_chord(self, ref_key: Key) -> str:
        if ref_key.mode == 'major':
            diatonic_chord = theory.diatonic_chords_in_major[ref_key.find_degree(spc=self.root).number]
        elif ref_key.mode == "natural_minor":
            diatonic_chord = theory.diatonic_chords_in_natural_minor[ref_key.find_degree(spc=self.root).number]
        elif ref_key.mode == "harmonic_minor":
            diatonic_chord = theory.diatonic_chords_in_harmonic_minor[ref_key.find_degree(spc=self.root).number]
        elif ref_key.mode == "melodic_minor":
            diatonic_chord = theory.diatonic_chords_in_melodic_minor[ref_key.find_degree(spc=self.root).number]
        else:
            raise NotImplementedError
        return diatonic_chord

    def numeral(self) -> str:
        numeral_match = theory.DCML_numeral_regex(self.numeral_string)
        result = numeral_match["numeral"]
        return result

    def chromatic_type(self) -> Literal["diatonic", "mixture", "tonicization"]:
        if "/" in self.numeral_string:
            return "tonicization"
        elif self.diatonic_chord(ref_key=self.local_key) == self.numeral():
            return "diatonic"
        else:
            raise NotImplementedError


def test1():
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/grieg_lyric_pieces/harmonies/op12n01.tsv',
        sep='\t')

    df_row = df.iloc[17]
    numeral = Numeral.from_df(df_row)
    result = numeral.non_diatonic_spcs(reference_key=numeral.global_key)
    print(f'{numeral=}')
    print(f'{result=}')


def test2():
    nc = NumeralChain.parse(numeral_str="ii/V", key=Key.from_string(key_str="C"))
    result = nc.key_if_tonicized()
    print(f'{result.tonic=}')
    degree = nc.degree()
    print(f'{degree=}')


def test3():
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/grieg_lyric_pieces/harmonies/op12n01.tsv',
        sep='\t')

    df_row = df.iloc[20]
    numeral = Numeral.from_df(df_row)
    print(f"{numeral=}")

    ref_key = numeral.key_if_tonicized()
    print(f'non-diatonic-notes: {numeral.non_diatonic_spcs(reference_key=ref_key)}')

    diatonic_chord = numeral.diatonic_chord(ref_key=numeral.local_key)
    print(f'{diatonic_chord=}')

    # index_m1 = metrics.pc_content_index_m1(numeral=numeral, reference_key=ref_key)
    # print(f'{index_m1=}')

    # new_pc = metrics.pc_content_index(numeral=numeral, nd_ref_key=ref_key, d5th_ref_tone=SpelledPitchClass("C"))
    # print(f'{new_pc=}')


if __name__ == '__main__':
    test3()
