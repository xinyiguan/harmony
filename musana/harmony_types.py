# Created by Xinyi Guan in 2022.
# Defining the harmony concepts.
from __future__ import annotations

import re
import typing
from abc import abstractmethod
from dataclasses import dataclass
from functools import cache

import regex_spm
from pitchtypes import SpelledPitchClass as _SpelledPitchClass
from pitchtypes import SpelledIntervalClass as SpelledIntervalClass

T = typing.TypeVar('T')


class SpelledPitchClass(_SpelledPitchClass):

    def alteration_ic(self) -> SpelledIntervalClass:
        if self.alteration() > 0:
            alteration_symbol = 'a' * self.alteration()
        elif self.alteration() == 0:
            alteration_symbol = 'P'  # perfect unison, no alterations
        elif self.alteration() < 0:
            alteration_symbol = 'd' * abs(self.alteration())
        else:
            raise ValueError(self.alteration)
        interval_class = SpelledIntervalClass(f'{alteration_symbol}1')
        return interval_class


class ProtocolHarmony(typing.Protocol):
    @abstractmethod
    def root(self) -> SpelledPitchClass:
        pass

    @abstractmethod
    def key_if_tonicized(self) -> Key:
        pass

    @abstractmethod
    def pc_set(self) -> typing.List[SpelledPitchClass]:
        pass

    @abstractmethod
    def chord_tones(self) -> typing.List[SpelledPitchClass]:
        pass


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

    def find_pc(self, degree: Degree) -> SpelledPitchClass:
        if self.mode == 'M':
            intervals = self._M_intervals
        elif self.mode == 'm':
            intervals = self._m_intervals
        else:
            raise ValueError(f'{self.mode=}')
        interval = intervals[degree.number - 1]
        pc = self.root + SpelledIntervalClass(interval)
        pc = SpelledPitchClass(pc.name())  # TODO: do it in the right way later (look up customized SpelledPitchClass)
        pc = pc - pc.alteration_ic()
        return pc

    @property
    @cache
    def pcs(self) -> typing.List[SpelledPitchClass]:
        return [self.find_pc(degree=Degree(number=n, alteration=0)) for n in range(1, 8)]

    def to_str(self) -> str:
        if self.mode == 'm':
            resulting_str = str(self.root).lower()
        elif self.mode == 'M':
            resulting_str = str(self.root)
        else:
            raise ValueError(f'not applicable mode')
        return resulting_str


@dataclass
class Degree:
    number: int
    alteration: int | bool  # when int: positive for "#", negative for "b", when bool: represent whether to use natural

    _numeral_scale_degree_dict = typing.ClassVar[{"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
                                                  "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}]

    _regex_arabic = re.compile("^((?P<modifiers>(b*)|(#*))?(?P<number>([0-9]+)))$")
    _regex_roman = re.compile("^(?P<modifiers>(b*)|(#*))(?P<roman_numeral>(IV|V?I{0,2}))$", re.I)

    def __add__(self, other: typing.Self) -> typing.Self:
        """
        n steps (0 steps is unison) <-- degree (1 is unison)
        |
        V
        n steps (0 steps is unison) --> degree (1 is unison)

        """
        number = ((self.number - 1) + (other.number - 1)) % 7 + 1
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    def __sub__(self, other: typing.Self) -> typing.Self:
        number = ((self.number - 1) - (other.number - 1)) % 7 + 1
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    @classmethod
    def parse(cls, degree_str: str) -> typing.Self:
        """
        Examples of arabic_degree: b7, #2, 3, 5, #5, ...
        Examples of scale degree: bV, bIII, #II, IV, vi, vii
        """
        match = regex_spm.fullmatch_in(degree_str)
        match match:
            case cls._regex_roman:
                degree_number = cls._numeral_scale_degree_dict.get(match['roman_numeral'])
            case cls._regex_arabic:
                degree_number = int(match['number'])
            case _:
                raise ValueError(f"could not match {match} with regex: {cls._regex_roman} or {cls._regex_arabic}")
        modifiers_match = match['modifiers']
        alteration = SpelledPitchClass(f'C{modifiers_match}').alteration()
        instance = cls(number=degree_number, alteration=alteration)
        return instance


@dataclass
class P:
    alt_steps: int


@dataclass
class IP:
    alt_steps: int

    def __post_init__(self):
        if self.alt_steps == 0:
            raise ValueError(f'{self.alt_steps=}')
        self.alt_steps = self.alt_steps


@dataclass
class HarmonyQuality:
    """ Examples:
    stack_of_thirds_3_Mm = HarmonyQuality(stacksize=3,ic_quality_list=[IP(1),IP(-1)]) # Major triad
    stack_of_thirds_4_mmm = HarmonyQuality(stacksize=3,ic_quality_list=[IP(-1),IP(-1),IP(-1)]) # fully diminished seventh chord
    stack_of_fifth_4_PPP = HarmonyQuality(stacksize=5,ic_quality_list=[P(0),P(0),P(0)])
    stack_of_fifth_3_ADP = HarmonyQuality(stacksize=5,ic_quality_list=[P(-1),P(1),P(0)])
    """

    stack_size: int
    interval_class_quality_list: typing.List[P | IP]


    _major_pattern = re.compile(r'(?P<major>I|II|III|IV|V|VI|VII)')
    _minor_pattern = re.compile(r'(?P<minor>i|ii|iii|iv|v|vi|vii)')
    _quality_pattern = re.compile(r'%|o|\+|M|\+M')
    _triad_pattern = re.compile(r'(?P<triad>6|64)')
    _tetrad_pattern = re.compile(r'(?P<tetrad>7|65|43|42|2)')

    @classmethod
    def parse_snp(cls, snp: SingleNumeralParts) -> typing.Self:  # current version: parse as stack of thirds
        """
        Examples of eligible strings:
        """
        rn = snp.roman_numeral if snp.roman_numeral else ''
        quality = snp.quality if snp.quality else ''
        figbass = snp.figbass if snp.figbass else ''
        assembled_snp = rn + quality + figbass

        match = regex_spm.fullmatch_in(assembled_snp)

        match match:
            case cls._major_pattern:  # major: Mm

                interval_class_quality_list = [IP(1), IP(-1)]
            case cls._minor_pattern:  # minor: mM
                interval_class_quality_list = [IP(-1), IP(1)]


            case _:
                raise ValueError()

        instance = cls(stack_size=3, interval_class_quality_list=interval_class_quality_list)
        return instance


@dataclass
class FiguredBass:
    figured_bass: typing.List[Degree]

    _figbass_degree_dict = {"7": [1, 3, 5, 7], "65": [3, 5, 7, 1], "43": [5, 7, 1, 3],
                            "42": [7, 1, 3, 5], "2": [7, 1, 3, 5],
                            "64": [5, 1, 3],
                            "6": [3, 5, 1]}  # assume the first number x is: root + x = bass , 1 := unison

    @classmethod
    def parse(cls, figbass_str: str) -> typing.Self:
        instance = cls(figured_bass=[Degree.parse(degree_str=str(x))
                                     for x in FiguredBass._figbass_degree_dict.get(figbass_str)])
        return instance


@dataclass
class AddedTones:
    added_tones: typing.List[Degree]

    _regex_pattern_1 = re.compile(r'(?P<plus>(\+))(?P<modifiers>[#b])?(?P<number>[246])')
    _regex_pattern_2 = re.compile(r'(?P<modifiers>[b#]*)(?P<number>9|11|13)')

    @classmethod
    def parse(cls, added_tones_str: str) -> typing.Self:
        match = regex_spm.match_in(added_tones_str)
        match match:
            case cls._regex_pattern_1:
                match_iter = re.finditer(cls._regex_pattern_1, string=added_tones_str)
            case cls._regex_pattern_2:
                match_iter = re.finditer(cls._regex_pattern_2, string=added_tones_str)
            case _:
                raise ValueError(
                    f"could not match {match} with regex: {cls._regex_pattern_1} or {cls._regex_pattern_2}")

        added_tones = [Degree.parse(degree_str=match[0]) for match in match_iter]
        instance = cls(added_tones=added_tones)
        return instance


@dataclass
class ReplacementTones:
    replacement_tones: typing.List[Degree]

    _regex_pattern = re.compile(r'(?P<modifiers>[#b])?(?P<number>[246])')

    @classmethod
    def parse(cls, replacement_tones_str: str) -> typing.Self:
        match_iter = re.finditer(cls._regex_pattern, string=replacement_tones_str)
        replacement_tones = [Degree.parse(degree_str=match[0]) for match in match_iter]
        instance = cls(replacement_tones=replacement_tones)
        return instance


@dataclass
class SingleNumeralParts:
    roman_numeral: str
    modifiers: str | None
    quality: str | None
    figbass: str | None
    added_tones: str | None
    replacement_tones: str | None

    # the regular expression conforms with the DCML annotation standards
    _sn_regex = re.compile("^(?P<modifiers>(b*)|(#*))?"  # accidentals
                           "(?P<roman_numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))"  # roman numeral
                           "(?P<quality>(%|o|\+|M|\+M))?"  # quality form
                           "(?P<figbass>(7|65|43|42|2|64|6))?"  # figured bass
                           "(\("
                           "((?P<added_tones>((\+)([#b])?([2-8]))+|(([#b])?(9|1[0-4]))+)?|"  # added tones, non-chord tones added within parentheses and preceded by a "+" or >8
                           "(?P<replacement_tones>(([#b])?([2-8]))+)?)"  # replaced chord tones expressed through intervals <= 8
                           "\))?$")

    @classmethod
    def parse(cls, numeral_str: str) -> typing.Self:
        # match with regex
        s_numeral_match = SingleNumeralParts._sn_regex.match(numeral_str)
        if s_numeral_match is None:
            raise ValueError(f"could not match '{numeral_str}' with regex: '{SingleNumeralParts._sn_regex.pattern}'")

        roman_numeral = s_numeral_match['roman_numeral']
        modifiers = s_numeral_match['modifiers'] if s_numeral_match['modifiers'] else None
        quality = s_numeral_match['quality'] if s_numeral_match['quality'] else None
        figbass = s_numeral_match['figbass'] if s_numeral_match['figbass'] else None
        added_tones = s_numeral_match['added_tones'] if s_numeral_match['added_tones'] else None
        replacement_tones = s_numeral_match['replacement_tones'] if s_numeral_match['replacement_tones'] else None

        instance = cls(modifiers=modifiers, roman_numeral=roman_numeral, quality=quality, figbass=figbass,
                       added_tones=added_tones, replacement_tones=replacement_tones)
        return instance


class SnpParsable(typing.Protocol):
    @classmethod
    @abstractmethod
    def parse_snp(cls, snp: SingleNumeralParts) -> typing.Self:
        pass


@dataclass(frozen=True)
class SingleNumeral(ProtocolHarmony):
    key: Key
    degree: Degree
    quality: HarmonyQuality  # 4 triad qualities and 6 seventh chord qualities
    figbass: FiguredBass | None
    added_tones: AddedTones | None
    replacement_tones: ReplacementTones | None

    @classmethod
    def parse(cls, key_str: str | Key, numeral_str: str) -> typing.Self:

        snp = SingleNumeralParts.parse(numeral_str=numeral_str)

        # parse key:
        if not isinstance(key_str, Key):
            key = Key.parse(key_str=key_str)
        else:
            key = key_str

        # parse quality, in stack of thirds: # TODO: rewrite this
        quality = HarmonyQuality.parse_snp(snp=snp)

        # parse degree:
        degree = Degree.parse(degree_str=snp.roman_numeral)

        # parse added_tones: # TODO: double check the annotation tutorial (advanced section) for more complex cases
        added_tones = AddedTones.parse(added_tones_str=snp.added_tones)

        # replacement_tones: # TODO: double check the annotation tutorial (advanced section) for more complex cases
        replacement_tones = ReplacementTones.parse(replacement_tones_str=snp.replacement_tones)

        # parse figbass:
        figbass = FiguredBass.parse(figbass_str=snp.figbass)

        # create class instance:
        instance = cls(key=key,
                       degree=degree, quality=quality, figbass=figbass,
                       added_tones=added_tones, replacement_tones=replacement_tones)
        return instance

    def root(self) -> SpelledPitchClass:

        root = self.key.find_pc(self.degree)

        return root

    def key_if_tonicized(self) -> Key:
        """
        Make the current numeral as the tonic, return the spelled pitch class of the root as Key.
        """
        key = Key(root=self.root(), mode=self.quality)  # TODO: quality as str or HarmonyQuality?
        return key

    def bass_degree(self) -> Degree:
        bass_degree = self.degree + self.figbass.figured_bass[0]  # TODO: set alias?
        return bass_degree

    def bass_pc(self) -> SpelledPitchClass:
        pc = self.key.find_pc(self.bass_degree())
        return pc

    def chord_tones(self) -> typing.List[SpelledPitchClass]:
        raise NotImplementedError

    def pc_set(self) -> typing.List[SpelledPitchClass]:
        pass


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

    def pc_set(self) -> typing.List[SpelledPitchClass]:
        pitchclass = self.chord_tones() | self.added_tones()
        return pitchclass

    def root(self) -> SpelledPitchClass:
        pass

    def key_if_tonicized(self) -> Key:
        pass

    def to_numeral(self) -> Numeral:
        pass

    def chord_tones(self) -> typing.List[SpelledPitchClass]:
        pass

    def added_tones(self) -> typing.List[SpelledPitchClass]:
        pass


if __name__ == '__main__':
    snp = SingleNumeralParts.parse('V')
    print(f'{snp=}')
    hq = HarmonyQuality.parse_snp(snp=snp)
    print(f'{hq=}')

    # ip = IP(0)
    # print(f'{ip=}')
    #
    # p = P(1)
    # print(f'{p=}')
