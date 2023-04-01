# Defining the harmony concepts.
from __future__ import annotations

import re
import typing
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pitchtypes
import regex_spm
from pitchtypes import SpelledPitchClass, SpelledIntervalClass

T = typing.TypeVar('T')


@dataclass
class Triad:
    root: SpelledPitchClass
    third: SpelledPitchClass
    fifth: SpelledPitchClass


class HarmonyRegexes:
    """a class to hold all the regular expressions in the harmony string input"""

    key_regex = re.compile("^(?P<class>[A-G])(?P<modifiers>(b*)|(#*))$", re.I)  # case-insensitive

    arabic_degree_regex = re.compile("^(?P<modifiers>[#b]*)(?P<number>(\d+))$")
    roman_degree_regex = re.compile("^(?P<modifiers>[#b]*)(?P<roman_numeral>([VI]+))$", re.I)

    figuredbass_regex = re.compile("(?P<figbass>(7|65|43|42|2|64|6))")

    added_tone_regex = re.compile(
        r'(\+[#b]*\d*)')  # TODO: ?? Added tones are always preceded by '+'?? double check the standards?

    replacement_tone_regex = re.compile(r'([\^v][#b]*\d*)')


@dataclass(frozen=True)
class Key:
    tonic: SpelledPitchClass
    mode: typing.Literal['major', 'minor']

    _MajorIntervals = pitchtypes.asic(things=np.array(['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7']))
    _NatrualMinorIntervals = pitchtypes.asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7']))
    _HarmonicMinorIntervals = pitchtypes.asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'M7']))

    @classmethod
    def parse(cls, key_str: str) -> typing.Self:

        if not isinstance(key_str, str):
            raise TypeError(f"expected string as input, got {key_str}")
        key_match = HarmonyRegexes.key_regex.fullmatch(key_str)
        if key_match is None:
            raise ValueError(f"could not match '{key_str}' with regex: '{HarmonyRegexes.key_regex.pattern}'")
        mode = 'major' if key_match['class'].isupper() else 'minor'
        tonic = SpelledPitchClass(key_match['class'].upper() + key_match['modifiers'])
        instance = cls(tonic=tonic, mode=mode)
        return instance

    @classmethod
    def transpose_to(cls, source_key: str,
                     by: SpelledIntervalClass,
                     target_mode: typing.Literal['major', 'minor']) -> typing.Self:

        source_key_cls = Key.parse(key_str=source_key)
        target_root: SpelledPitchClass = source_key_cls.tonic + by
        instance = cls(tonic=target_root, mode=target_mode)
        return instance

    @property
    def parallel(self) -> Key:
        raise NotImplementedError

    @property
    def relative(self) -> Key:
        if self.mode == 'major':
            new_tonic = self.tonic - SpelledIntervalClass('m3')
            relative_minor = Key(tonic=new_tonic, mode='minor')
            return relative_minor
        else:
            new_tonic = self.tonic + SpelledIntervalClass('m3')
            relative_major = Key(tonic=new_tonic, mode='major')
            return relative_major

    def scale(self) -> typing.List[SpelledPitchClass]:
        the_scale = [self.find_pc(Degree.parse(str(i))) for i in range(1, 8)]
        return the_scale

    @property
    def accidentals(self) -> int:
        return abs(self.relative.tonic.fifths()) if self.mode == 'minor' else abs(self.tonic.fifths())

    def find_pc(self, degree: Degree) -> SpelledPitchClass:
        if self.mode == 'major':
            intervals = self._MajorIntervals
        elif self.mode == 'minor':
            intervals = self._NatrualMinorIntervals
        else:
            raise ValueError(f'{self.mode=}')

        if degree.alteration > 0:
            alteration_symbol = 'a' * degree.alteration
        elif degree.alteration == 0:
            alteration_symbol = 'P'  # perfect unison, no alterations
        elif degree.alteration < 0:
            alteration_symbol = 'd' * abs(degree.alteration)
        else:
            raise ValueError(degree.alteration)
        interval_alteration = SpelledIntervalClass(f'{alteration_symbol}1')

        interval = intervals[degree.number - 1] + interval_alteration
        pc = self.tonic + interval

        return pc

    def find_sd(self, pc: SpelledPitchClass) -> Degree:

        if pc.alteration() > 0:
            alteration_symbol = 'a' * pc.alteration()
        elif pc.alteration() == 0:
            alteration_symbol = 'P'  # perfect unison, no alterations
        elif pc.alteration() < 0:
            alteration_symbol = 'd' * abs(pc.alteration())
        else:
            raise ValueError(pc.alteration)
        interval_class = SpelledIntervalClass(f'{alteration_symbol}1')

        sd_alteration = interval_class.alteration()

        sd_integer = [pc.letter() for pc in self.pcs].index(pc.letter()) + 1
        sd = Degree(number=sd_integer, alteration=sd_alteration)
        return sd

    @property
    def pcs(self) -> typing.List[SpelledPitchClass]:
        return [self.find_pc(degree=Degree(number=n, alteration=0)) for n in range(1, 8)]

    def to_str(self) -> str:
        if self.mode == 'minor':
            resulting_str = str(self.tonic).lower()
        elif self.mode == 'major':
            resulting_str = str(self.tonic)
        else:
            raise ValueError(f'not applicable mode')
        return resulting_str


@dataclass
class Degree:
    number: int
    alteration: int | bool  # when int: positive for "#", negative for "b", when bool: represent whether to use natural

    _numeral_scale_degree_dict = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
                                  "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}

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
            case HarmonyRegexes.roman_degree_regex:
                degree_number = cls._numeral_scale_degree_dict.get(match['roman_numeral'])
            case HarmonyRegexes.arabic_degree_regex:
                degree_number = int(match['number'])
            case _:
                raise ValueError(
                    f"could not match {match} with regex: {HarmonyRegexes.roman_degree_regex} or {HarmonyRegexes.arabic_degree_regex}")
        modifiers_match = match['modifiers']
        alteration = SpelledPitchClass(f'C{modifiers_match}').alteration()
        instance = cls(number=degree_number, alteration=alteration)
        return instance

    def to_sic(self) -> SpelledIntervalClass:
        sic = ...
        return sic

    def to_spc(self) -> SpelledPitchClass:
        """relative to C"""
        spc = Key.parse(key_str='C').find_pc(degree=self)
        return spc


class IntervalQuality:
    @classmethod
    @abstractmethod
    def from_interval_class(cls, spelled_interval_class: SpelledIntervalClass) -> typing.Self:
        pass


@dataclass
class P(IntervalQuality):
    """
    This is a type class for perfect intervals
    alt_steps = 0 means e.g. perfect unison, perfect fifths
    alt_steps = 1 means aug 1 (A1), aug 5(A5), aug 4...
    alt_steps = -1 means dim 1 (D1) ...
    """

    alt_steps: int

    @classmethod
    def from_interval_class(cls, spelled_interval_class: SpelledIntervalClass) -> typing.Self:  # TODO: fill in
        pass


@dataclass
class IP(IntervalQuality):
    """
    This is a type class for imperfect intervals
    alt_steps = 1 means M2, M3, M6 ...
    alt_steps = 2 means a2, a3, a6 ...
    alt_steps = 3 means aa2, aa3, ...
    alt_steps = -1 means m2, m3, m6 ...
    """
    alt_steps: int

    def __post_init__(self):
        if self.alt_steps == 0:
            raise ValueError(f'{self.alt_steps=}')
        self.alt_steps = self.alt_steps

    @classmethod
    def from_interval_class(cls, spelled_interval_class: SpelledIntervalClass) -> typing.Self:  # TODO: fill in
        # check if input makes sense:

        alt_steps = ...

        instance = cls(alt_steps=alt_steps)
        return instance

    def to_interval_class(self) -> SpelledIntervalClass:
        raise NotImplementedError


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

    _map_to_ic_qualities_3 = {(True, '+'): [1, 1],  # augmented triad
                              (True, ''): [1, -1],  # major triad
                              (False, ''): [-1, 1],  # minor triad
                              (False, 'o'): [-1, -1]}  # diminished triad

    _map_to_ic_qualities_4 = {(True, '+M'): [1, 1, 1],  # aug major 7th
                              (False, 'M'): [-1, 1, 1],  # minor major 7th
                              (True, 'M'): [1, -1, 1],  # major major 7th
                              (True, '+'): [1, 1, -1],  # aug minor 7th
                              (False, '%'): [-1, -1, 1],  # half dim 7th
                              (False, ''): [-1, 1, -1],  # minor minor 7th
                              (True, ''): [1, -1, -1],  # dominant 7th
                              (False, 'o'): [-1, -1, -1]}  # fully dim 7th

    _which_dict = {3: _map_to_ic_qualities_3,
                   4: _map_to_ic_qualities_4}

    @classmethod
    def smart_init(cls, n_chord_tones: int, upper: bool,
                   form_symbol: typing.Literal['o', '+', '%', 'M', '+M', '']) -> typing.Self:
        dict_to_check = cls._which_dict[n_chord_tones]
        ic_quality_list = [IP(x) for x in dict_to_check[(upper, form_symbol)]]
        instance = cls(stack_size=3, interval_class_quality_list=ic_quality_list)
        return instance

    @property
    def major_minor_mode(self) -> typing.Literal['major', 'minor']:
        mode = 'major' if (
                self.interval_class_quality_list[0] == IP(1) and self.interval_class_quality_list[
            1] == IP(-1)) else 'minor'
        return mode

    def to_sic(self, root: SpelledPitchClass) -> typing.List[SpelledIntervalClass]:
        raise NotImplementedError


@dataclass
class FiguredBass:
    degrees: typing.List[Degree]

    _figbass_degree_dict = {"7": [1, 3, 5, 7], "65": [3, 5, 7, 1], "43": [5, 7, 1, 3],
                            "42": [7, 1, 3, 5], "2": [7, 1, 3, 5],
                            "64": [5, 1, 3],
                            "6": [3, 5, 1]}  # assume the first number x is: root + x = bass , 1 := unison

    @classmethod
    def parse(cls, figbass_str: str) -> typing.Self:  # TODO: revise with alteration, add a dict to parse
        match = regex_spm.fullmatch_in(figbass_str)
        match match:
            case HarmonyRegexes.figuredbass_regex:
                instance = cls(degrees=[Degree.parse(degree_str=str(x))
                                        for x in FiguredBass._figbass_degree_dict.get(figbass_str)])
            case _:  # otherwise, assume root position triad
                instance = cls(degrees=[Degree(number=1, alteration=0),
                                        Degree(number=3, alteration=0),
                                        Degree(number=5, alteration=0)])
        return instance

    def n_chord_tones(self) -> int:
        return len(self.degrees)


@dataclass
class AddedTones:
    degrees: typing.List[Degree]

    @classmethod
    def parse(cls, added_tones_str: str) -> typing.Self:
        match = regex_spm.match_in(added_tones_str)

        match match:
            case HarmonyRegexes.added_tone_regex:
                match_iter = re.finditer(HarmonyRegexes.added_tone_regex, string=added_tones_str)
                degrees = [Degree.parse(degree_str=match[0].replace('+', '')) for match in match_iter]
                instance = cls(degrees=degrees)
                return instance
            case _:
                return None


@dataclass
class ReplacementTones:
    degrees: typing.List[Degree]

    _regex_pattern = re.compile(r'([\^v][#b]*\d*)')

    @classmethod
    def parse(cls, replacement_tones_str: str) -> typing.Self:
        """
        V7(9) , V7(b9) , V7(v#9), V7(^9) , V7(^b9) , V7(#9),
        """
        match = regex_spm.match_in(replacement_tones_str)
        match match:
            case cls._regex_pattern:
                match_iter = re.finditer(cls._regex_pattern, string=replacement_tones_str)
            case _:
                return None
        replacement_tones = [Degree.parse(degree_str=match[0].replace('^', '').replace('v', '')) for match in
                             match_iter]  # TODO: current version do not consider replaced from above or below (i.e., 'v', '^')
        instance = cls(degrees=replacement_tones)
        return instance


@dataclass
class SingleNumeralParts:
    roman_numeral: str
    modifiers: str
    form: str
    figbass: str
    added_tones: str
    replacement_tones: str

    # the regular expression conforms with the DCML annotation standards
    # _sn_regex = re.compile("^(?P<modifiers>(b*)|(#*))"  # accidentals
    #                        "(?P<roman_numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))"  # roman numeral
    #                        "(?P<form>(%|o|\+|M|\+M))?"  # quality form
    #                        "(?P<figbass>(7|65|43|42|2|64|6))?"  # figured bass
    #                        "(?P<added_tones>(\(\+[#b]*\d*)\))?"  # added tones, non-chord tones added within parentheses and preceded by a "+"
    #                        "(?P<replacement_tones>(\([\^v]*[#b]*\d*)\))?$")  # replaced chord tones expressed through intervals <= 8

    _sn_regex = re.compile("^(?P<modifiers>(b*)|(#*))"  # accidentals
                           "(?P<roman_numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))"  # roman numeral
                           "(?P<form>(%|o|\+|M|\+M))?"  # quality form
                           "(?P<figbass>(7|65|43|42|2|64|6))?"  # figured bass
                           "(\()?"
                           "(?P<replacement_tones>([\^v]*[#b]*\d*))?"
                           "(?P<added_tones>((\+[#b]*\d)*))?"  # added tones, non-chord tones added within parentheses and preceded by a "+" 
                           "\)?$")  # replaced chord tones expressed through intervals <= 8

    @classmethod
    def parse(cls, numeral_str: str) -> typing.Self:
        # match with regex
        s_numeral_match = SingleNumeralParts._sn_regex.match(numeral_str)
        if s_numeral_match is None:
            raise ValueError(f"could not match '{numeral_str}' with regex: '{SingleNumeralParts._sn_regex.pattern}'")

        roman_numeral = s_numeral_match['roman_numeral']
        modifiers = s_numeral_match['modifiers'] if s_numeral_match['modifiers'] else ''
        form = s_numeral_match['form'] if s_numeral_match['form'] else ''
        figbass = s_numeral_match['figbass'] if s_numeral_match['figbass'] else ''
        added_tones = s_numeral_match['added_tones'].replace('(', '').replace(')', '') if s_numeral_match[
            'added_tones'] else ''
        replacement_tones = s_numeral_match['replacement_tones'].replace('(', '').replace(')', '') if s_numeral_match[
            'replacement_tones'] else ''

        instance = cls(modifiers=modifiers, roman_numeral=roman_numeral, form=form, figbass=figbass,
                       added_tones=added_tones, replacement_tones=replacement_tones)
        return instance


class SnpParsable(typing.Protocol):
    @classmethod
    @abstractmethod
    def parse_snp(cls, snp: SingleNumeralParts) -> typing.Self:
        pass


@dataclass
class Chain(typing.Generic[T]):
    head: T
    tail: typing.Optional[Chain[T]]


class AbstractHarmony(typing.Protocol):
    @abstractmethod
    def root(self) -> SpelledPitchClass:
        pass

    @abstractmethod
    def key_if_tonicized(self) -> Key:
        pass


@dataclass(frozen=True)
class SingleNumeral(AbstractHarmony):
    key: Key
    degree: Degree
    quality: HarmonyQuality
    figbass: FiguredBass
    pcs: typing.List[SpelledPitchClass]

    @classmethod
    def parse(cls, key_str: str | Key, numeral_str: str) -> typing.Self:

        snp = SingleNumeralParts.parse(numeral_str=numeral_str)

        # parse key:
        if not isinstance(key_str, Key):
            key = Key.parse(key_str=key_str)
        else:
            key = key_str

        # parse degree:
        degree = Degree.parse(degree_str=snp.modifiers + snp.roman_numeral)

        # parse added_tones: # TODO: double check the annotation tutorial (advanced section) for more complex cases
        added_tones = AddedTones.parse(added_tones_str=snp.added_tones)

        # replacement_tones: # TODO: double check the annotation tutorial (advanced section) for more complex cases
        replacement_tones = ReplacementTones.parse(replacement_tones_str=snp.replacement_tones)

        # parse figbass:
        figbass = FiguredBass.parse(figbass_str=snp.figbass)

        # parse quality, in stack of thirds:
        quality = HarmonyQuality.smart_init(n_chord_tones=figbass.n_chord_tones(), upper=snp.roman_numeral.isupper(),
                                            form_symbol=snp.form)

        # create class instance:
        instance = cls(key=key,
                       degree=degree, quality=quality, figbass=figbass,
                       pcs=...)
        return instance

    def root(self) -> SpelledPitchClass:
        root = self.key.find_pc(self.degree)
        return root

    def key_if_tonicized(self) -> Key:
        mode = self.quality.major_minor_mode
        key = Key(tonic=self.root(), mode=mode)
        return key

    def bass_degree(self) -> Degree:
        bass_degree = self.degree + self.figbass.degrees[0]
        return bass_degree

    def bass_pc(self) -> SpelledPitchClass:
        pc = self.key.find_pc(self.bass_degree())
        return pc

    def pcs(self) -> typing.List[SpelledPitchClass]:
        raise NotImplementedError


@dataclass
class Numeral(Chain[SingleNumeral]):
    key: Key

    @classmethod
    def parse(cls, key_str: str | Key, numeral_str: str) -> typing.Self:
        # numeral_str examples: "#ii/V", "##III/bIV/V", "bV", "IV(+6)", "vii%7/IV"

        if "/" in numeral_str:
            L_numeral_str, R_numeral_str = numeral_str.split("/", maxsplit=1)
            tail = cls.parse(key_str=key_str, numeral_str=R_numeral_str)
            head = SingleNumeral.parse(key_str=tail.head.key_if_tonicized(), numeral_str=L_numeral_str)

        else:
            head = SingleNumeral.parse(key_str=key_str, numeral_str=numeral_str)
            tail = None
        key = Key.parse(key_str=key_str) if isinstance(key_str, str) else key_str
        instance = cls(head=head, tail=tail, key=key)
        return instance

    def key_if_tonicized(self) -> Key:
        result_key = self.head.key_if_tonicized()
        return result_key


@dataclass(frozen=True)
class TonalHarmony(AbstractHarmony):
    globalkey: Key
    localkey: Key
    numeral: Numeral
    pcs: typing.List[SpelledPitchClass]

    @classmethod
    def parse_string(cls, globalkey_str: str, localkey_numeral_str: str, chord_str: str) -> typing.Self:
        # chord_str examples: "IV(+6)", "vii%7/IV", "ii64"
        globalkey = Key.parse(key_str=globalkey_str)
        localkey = Numeral.parse(key_str=globalkey_str, numeral_str=localkey_numeral_str).head.key_if_tonicized()
        compound_numeral = Numeral.parse(key_str=localkey.to_str(), numeral_str=chord_str)
        pcs = ...
        instance = cls(globalkey=globalkey, localkey=localkey, numeral=compound_numeral, pcs=pcs)
        return instance
    @staticmethod
    def convert_added_tones_to_5th(entry: float | str | pd.NA) -> typing.List[int]:

        if pd.isna(entry):
            result = []
        elif isinstance(entry,float):
            result = [int(entry)]
        elif isinstance(entry,str):
            result = list(map(int, entry.split(',')))
        else:
            raise TypeError(entry)
        print(f'{entry=} {result=}')
        return result
    @classmethod
    def from_df_row(cls, df_row: pd.DataFrame) -> typing.Self:
        GK = Key.parse(key_str=df_row['globalkey'])
        LK = SingleNumeral.parse(key_str=GK, numeral_str=df_row['localkey']).key_if_tonicized()
        numeral = Numeral.parse(key_str=LK, numeral_str=df_row['chord'])

        chord_tones_in_5th = list(map(int, df_row['chord_tones'].split(",")))
        chord_tones_pc = [SpelledPitchClass.from_fifths(fifths=x) for x in chord_tones_in_5th]
        added_tones_in_5th = cls.convert_added_tones_to_5th(df_row['added_tones'])
        print(f'{added_tones_in_5th=}')
        #added_tones_in_5th = list(map(int, df_row['added_tones'].split(",")))
        added_tones_pc = [SpelledPitchClass.from_fifths(fifths=x) for x in added_tones_in_5th]

        pcs = chord_tones_pc + added_tones_pc
        instance = cls(globalkey=GK,
                       localkey=LK,
                       numeral=numeral,
                       pcs=pcs)
        return instance

    def root(self) -> SpelledPitchClass:
        result = self.numeral.head.root()
        return result

    def third(self) -> SpelledPitchClass:
        scale_mode = self.numeral.head.quality.major_minor_mode
        if scale_mode == 'major':
            ic_to_add = SpelledIntervalClass('M3')
        else:
            ic_to_add = SpelledIntervalClass('m3')
        result = self.numeral.head.root() + ic_to_add
        return result

    def fifth(self) -> SpelledPitchClass:
        the_root = self.root()
        result = the_root + SpelledIntervalClass('P5')
        return result

    # def key_if_tonicized(self, ref_key: typing.Literal['global', 'local', 'chord']) -> Key:
    #     """with different reference tonic: global key, local key or the chord level """
    #     match ref_key:
    #         case "global":
    #             result_key_root = self.numeral.head.root()
    #
    #         case "local":
    #             ...
    #         case "chord":
    #             result_key_root = self.numeral.key_if_tonicized()
    #
    #     result_key_mode = 'major' if (self.numeral.head.quality.interval_class_quality_list[0] == IP(1) and
    #                                   self.numeral.head.quality.interval_class_quality_list[1] == IP(-1)) else 'minor'
    #     result_key = Key(tonic=result_key_root, mode=result_key_mode)
    #     return result_key

    def key_if_tonicized(self) -> Key:
        raise NotImplementedError

def test():
    df: pd.DataFrame = pd.read_csv(
        '/Users/xinyiguan/MusicData/dcml_corpora/debussy_suite_bergamasque/harmonies/l075-01_suite_prelude.tsv',
        sep='\t')

    df_row = df.iloc[73]

    result = TonalHarmony.from_df_row(df_row)
    print(f'{result=}')

if __name__ == '__main__':
    test()
