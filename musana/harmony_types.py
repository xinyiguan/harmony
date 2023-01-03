# Created by Xinyi Guan in 2022.
# Defining the harmony concepts.
from __future__ import annotations

import re
import typing
from abc import abstractmethod
from dataclasses import dataclass

import pandas as pd
from pitchtypes import SpelledPitchClass, SpelledIntervalClass


@dataclass(frozen=True)
class Key:
    root: SpelledPitchClass
    mode: typing.Literal['M', 'm']

    _M_intervals = ['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7']
    _m_intervals = ['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7']

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
        if self.mode == 'M':
            intervals = self._M_intervals
        elif self.mode == 'm':
            intervals = self._m_intervals
        else:
            raise ValueError(f'{self.mode=}')
        interval = intervals[degree - 1]
        pc = self.root + SpelledIntervalClass(interval)
        return pc



    def pcs(self) -> typing.List[SpelledPitchClass]:
        return [self.find_pc(degree=n) for n in range(1, 8)]

    def to_str(self) -> str:
        if self.mode == 'm':
            resulting_str = str(self.root).lower()
        elif self.mode == 'M':
            resulting_str = str(self.root)
        else:
            raise ValueError(f'not applicable mode')
        return resulting_str


class ProtocolHarmony(typing.Protocol):
    @abstractmethod
    def root(self) -> SpelledPitchClass:
        pass

    @abstractmethod
    def key_if_tonicized(self) -> Key:
        pass

    @abstractmethod
    def pc_set(self) -> typing.Set[SpelledPitchClass]:
        pass

    @abstractmethod
    def chord_tones(self) -> typing.Set[SpelledPitchClass]:
        pass

    @abstractmethod
    def added_tones(self) -> typing.Set[SpelledPitchClass]:
        pass

class Degree:
    number: int
    alteration: int

    def __add__(self) -> typing.Self:
        raise NotImplementedError

@dataclass(frozen=True)
class SingleNumeral(ProtocolHarmony):
    key: Key
    degree: Degree
    quality: typing.Literal['M', 'm']
    form: typing.Optional[typing.Literal["%", "o", "M", "m"]]
    figbass: Degree
    added_tones:typing.List[Degree]
    replacement_tones:typing.List[Degree]

    # the regular expression conforms with the DCML annotation standards

    _sn_regex = re.compile(
        "^(?P<modifiers>(b*)|(#*))?(?P<roman_numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))(?P<form>%|o|\+|M|\+M)?(?P<figbass>7|65|43|42|2|64|6)?(?:\((?P<changes>(?:[\+-\^v]?[b#]*\d)+)\))?$")

    @classmethod
    def parse(cls, key_str: str | Key, numeral_str: str) -> SingleNumeral:
        numeral_scale_degree_dict = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
                                     "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}

        # match with regex
        s_numeral_match = SingleNumeral._sn_regex.match(numeral_str)
        if s_numeral_match is None:
            raise ValueError(
                f"could not match '{numeral_str}' with regex: '{SingleNumeral._sn_regex.pattern}'")
        # parse key
        if not isinstance(key_str, Key):
            key = Key.parse(key_str=key_str)
        else:
            key = key_str

        # parse degree, quality, alterations
        degree = numeral_scale_degree_dict.get(s_numeral_match['roman_numeral'])  # TODO: account for Ger/Fr/It
        quality = "M" if s_numeral_match['roman_numeral'].isupper() else "m"

        modifiers_count = len(s_numeral_match['modifiers'])
        if all((char == "#" for char in s_numeral_match['modifiers'])):
            alteration = modifiers_count
        elif all((char == "b" for char in s_numeral_match['modifiers'])):
            alteration = -modifiers_count
        else:
            raise ValueError(f'Unexpected mixed accidentals')

        # create class instance:
        instance = cls(key=key, roman_numeral=s_numeral_match['roman_numeral'],
                       degree=degree, alteration=alteration, quality=quality,
                       form=s_numeral_match['form'], figbass=s_numeral_match['figbass'],
                       changes=s_numeral_match['changes'])
        return instance

    def root(self) -> SpelledPitchClass:

        scale_pitch = self.key.find_pc(self.degree)

        if self.alteration > 0:
            alteration_symbol = 'a' * self.alteration
        elif self.alteration == 0:
            alteration_symbol = 'P'  # perfect unison, no alterations
        elif self.alteration < 0:
            alteration_symbol = 'd' * abs(self.alteration)
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

    def chord_tones(self) -> typing.List[SpelledPitchClass]:
        raise NotImplementedError

    # Todo: added_tones should be a field, not method.
    # def added_tones(self) -> typing.List[SpelledPitchClass]:
    #     # added_tone "reflects only those non-chord tones that were added using, within parentheses,
    #     # intervals preceded by + or/and greater than 8."
    #
    #     if self.changes and ("+" in self.changes or 8 < int(self.changes) < 14):
    #         added_tones = [x for x in self.changes.split("+")[1:]]
    #
    #     else:
    #         added_tones = []
    #     return added_tones

    def pc_set(self) -> typing.Set[SpelledPitchClass]:
        pass


T = typing.TypeVar('T')


@dataclass
class Chain(typing.Generic[T]):
    head: T
    tail: typing.Optional[Chain[T]]


class Numeral(Chain[SingleNumeral]):
    numeral_regex = re.compile(
        "^(?P<numeral>[b#]*(?:VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))(?P<form>%|o|\+|M|\+M)?(?P<figbass>7|65|43|42|2|64|6)?(?:\((?P<changes>(?:[\+-\^v]?[b#]*\d)+)\))?(?:/(?P<relativeroot>(?:[b#]*(?:VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?$")

    @classmethod
    def parse(cls, key_str: str, numeral_str: str) -> typing.Self:
        # numeral_str examples: "#ii/V", "##III/bIV/V", "bV", "IV(+6)", "vii%7/IV"

        if "/" in numeral_str:
            L_numeral_str, R_numeral_str = numeral_str.split("/", maxsplit=1)
            R = cls.parse(key_str=key_str, numeral_str=R_numeral_str)
            # todo: makesure the key works
            L = SingleNumeral.parse(key_str=R.head.key_if_tonicized(), numeral_str=L_numeral_str)

        else:
            L = SingleNumeral.parse(key_str=key_str, numeral_str=numeral_str)
            R = None

        instance = cls(head=L, tail=R)
        return instance


@dataclass(frozen=True)
class TonalHarmony(ProtocolHarmony):
    globalkey: Key
    numeral: Numeral
    bookeeping: typing.Dict[str, str]  # for bookkeeping

    @classmethod
    def parse(cls, globalkey_str: str, localkey_numeral_str: str, chord_str: str) -> typing.Self:
        # chord_str examples: "IV(+6)", "vii%7/IV", "ii64"
        globalkey = Key.parse(key_str=globalkey_str)
        localkey = Numeral.parse(key_str=globalkey_str, numeral_str=localkey_numeral_str).head.key_if_tonicized()
        compound_numeral = Numeral.parse(key_str=localkey.to_str(), numeral_str=chord_str)
        instance = cls(globalkey=globalkey, numeral=compound_numeral,
                       bookeeping={'globalkey_str': globalkey_str,
                                   'localkey_numeral_str': localkey_numeral_str,
                                   'chord_str': chord_str,
                                   })
        return instance

    def pc_set(self) -> typing.Set[SpelledPitchClass]:
        pitchclass = self.chord_tones() | self.added_tones()
        return pitchclass

    def root(self) -> SpelledPitchClass:
        pass

    def key_if_tonicized(self) -> Key:
        pass

    def to_numeral(self) -> Numeral:
        pass

    def chord_tones(self) -> typing.Set[SpelledPitchClass]:
        pass

    def added_tones(self) -> typing.Set[SpelledPitchClass]:
        pass


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
    roots: typing.List[str]
    roman_numerals: typing.List[str]
    chords: typing.List[str]

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
    # tonal_harmony = TonalHarmony.parse(globalkey_str='F', localkey_numeral_str='V', chord_str='bIII/vi')
    # print(tonal_harmony)
    #
    # sn = SingleNumeral.parse(key_str='d', numeral_str="ii(+4+#2)")
    # pc_in_scale = sn.key_if_tonicized().pcs()
    # print('em scale:: ', pc_in_scale)
    # triad_tones = [pc_in_scale[i] for i in (0, 2, 4)]

    key = Key.parse(key_str="d")

