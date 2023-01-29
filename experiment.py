import re
import typing
from dataclasses import dataclass

import numpy as np
import regex_spm
from pitchtypes import SpelledPitchClass, SpelledIntervalClass


@dataclass
class Degree:
    number: int
    alteration: int | bool  # when int: positive for "#", negative for "b", when bool: represent whether to use natural

    _numeral_scale_degree_dict = typing.ClassVar[{"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
                                                  "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}]

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


class SingleNumeralRegex:
    modifiers = re.compile("(b*)|(#*)?")  # accidentals
    roman_numeral = re.compile("VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none")  # roman numeral
    form = re.compile("(%|o|\+|M|\+M)?")  # form
    figbass = re.compile("(7|65|43|42|2|64|6)?")  # figured bass
    added_tones = re.compile(
        "((\+)([#b])?([2-8]))+|([#b])?(9|1[0-4])")  # added tones, non-chord tones added within parentheses and preceded by a "+" or >8
    replacement_tones = re.compile("([#b])?([2-8])+")  # replaced chord tones expressed through intervals <= 8


def test():
    _sn_regex = re.compile(
        "^(?P<modifiers>(b*)|(#*))?"  # accidentals
        "(?P<roman_numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))"  # roman numeral
        "(?P<form>(%|o|\+|M|\+M))?"  # form
        "(?P<figbass>(7|65|43|42|2|64|6))?"  # figured bass
        "(\("
        "((?P<added_tones>((\+)([#b])?([2-8]))+|([#b])?(9|1[0-4]))?|"  # added tones, non-chord tones added within parentheses and preceded by a "+" or >8
        "(?P<replacement_tones>(([#b])?([2-8]))+)?)"  # replaced chord tones expressed through intervals <= 8
        "\))?$")

    # quality= ['M', 'm', '%', 'o', '+', '7', 'M7', 'm7', '%7', 'o7', '+7']

    numeral_str = "bII6"

    s_numeral_match = _sn_regex.match(numeral_str)
    print(f'{s_numeral_match=}')

    # def regex_matching_condition(group_name):
    #     return s_numeral_match.get(group_name, '')


def test_regex_spm():
    match regex_spm.fullmatch_in("123,45"):
        case r"(\d+),(?P<second>\d+)" as m:
            print("Notice the `as m` at the end of the line above")
            print(f"The first group is {m[1]}")
            print(f"The second group is {m['second']}")
            print(f"The full `re.Match` object is available as {m.match}")


@dataclass
class SingleNumeralParts:
    modifiers: str
    roman_numeral: str
    form: str
    figbass: str
    added_tones: str
    replacement_tones: str

    # the regular expression conforms with the DCML annotation standards
    _sn_regex = re.compile("^(?P<modifiers>(b*)|(#*))?"  # accidentals
                           "(?P<roman_numeral>(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none))"  # roman numeral
                           "(?P<form>(%|o|\+|M|\+M))?"  # form
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
        modifiers = s_numeral_match['modifiers'] if s_numeral_match['modifiers'] else ''
        form = s_numeral_match['form'] if s_numeral_match['form'] else ''
        figbass = s_numeral_match['figbass'] if s_numeral_match['figbass'] else ''
        added_tones = s_numeral_match['added_tones'] if s_numeral_match['added_tones'] else ''
        replacement_tones = s_numeral_match['replacement_tones'] if s_numeral_match['replacement_tones'] else ''

        instance = cls(modifiers=modifiers, roman_numeral=roman_numeral, form=form, figbass=figbass,
                       added_tones=added_tones, replacement_tones=replacement_tones)
        return instance


@dataclass
class P:
    alt_steps: int


@dataclass
class IP:
    def __init__(self, alt_steps: int):
        if alt_steps == 0:
            raise ValueError(f'{alt_steps=}')
        self.alt_steps = alt_steps

class HarmonyRegexes:
    """a class to hold all the regular expressions in the harmony string input"""

    key_regex = re.compile("^(?P<class>[A-G])(?P<modifiers>(b*)|(#*))$", re.I)  # case-insensitive

    arabic_degree_regex = re.compile("^((?P<modifiers>(b*)|(#*))?(?P<number>([0-9]+)))$")
    roman_degree_regex = re.compile("^(?P<modifiers>(b*)|(#*))?(?P<roman_numeral>(IV|V?I{0,2}))$", re.I)

    figuredbass_regex = re.compile(r'(?P<figbass>(7|65|43|42|2|64|6))')

    added_tone_regex = re.compile(r'((?P<plus>\+)(?P<modifiers>(#)*|(b)*)?(?P<number>\d*))+')   # TODO: ?? Added tones are always preceded by '+'?? #TODO: double check the standards?





if __name__ == '__main__':
    roman_degree_regex = re.compile("^(?P<modifiers>(#)*|(b)*)?(?P<roman_numeral>([VI]+))$", re.I)

    result = roman_degree_regex.match(string='#vi')
    print(f'{result=}')