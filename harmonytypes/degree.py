from __future__ import annotations

import re
from typing import Self, Literal
from dataclasses import dataclass

import regex_spm
from pitchtypes import SpelledPitchClass, SpelledIntervalClass

from harmonytypes import theory
from harmonytypes.theory import intervals_in_key_dict


@dataclass
class Degree:
    number: int
    alteration: int  # sign (+/-) indicates direction (♯/♭), value indicates number of signs added

    _roman_degree_regex = re.compile("^(?P<modifiers>[#b]*)(?P<roman_numeral>([VI]+))$", re.I)
    _arabic_degree_regex = re.compile("^(?P<modifiers>[#b]*)(?P<number>(\d+))$")

    def __repr__(self):
        if self.alteration == 0:
            sign = ''
        elif self.alteration > 0:
            sign = "#" * abs(self.alteration)
        elif self.alteration < 0:
            sign = "b" * abs(self.alteration)
        else:
            raise ValueError(f'invalid {self.alteration=}')
        return f'{sign}{self.number}'

    def __add__(self, other: Self) -> Self:
        """
        n steps (0 steps is unison) <-- degree (1 is unison)
        |
        V
        n steps (0 steps is unison) --> degree (1 is unison)

        """
        number = ((self.number - 1) + (other.number - 1)) % 7 + 1
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    def __sub__(self, other: Self) -> Self:
        number = ((self.number - 1) - (other.number - 1)) % 7 + 1
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    @classmethod
    def from_string(cls, degree_str: str) -> Self:
        """
        Examples of arabic_degree: b7, #2, 3, 5, #5, ...
        Examples of scale degree: bV, bIII, #II, IV, vi, vii
        """

        match = regex_spm.fullmatch_in(degree_str)
        match match:
            case cls._roman_degree_regex:
                degree_number = theory.roman_numeral_scale_degree_dict.get(match['roman_numeral'])
            case cls._arabic_degree_regex:
                degree_number = int(match['number'])
            case _:
                raise ValueError(
                    f"could not match {match} with regex: {cls._roman_degree_regex} or {cls._arabic_degree_regex}")
        modifiers_match = match['modifiers']
        alteration = SpelledPitchClass(f'C{modifiers_match}').alteration()
        instance = cls(number=degree_number, alteration=alteration)
        return instance

    @classmethod
    def from_sic(cls, sic: SpelledIntervalClass, mode=Literal["major", "natural_minor"]) -> Self:

        if mode == 'major':
            intervals = intervals_in_key_dict['major']

        elif mode == 'natural_minor':
            intervals = intervals_in_key_dict['natural_minor']

        else:
            raise ValueError(f'Invalid mode {mode}')

        number = sic.degree() + 1
        alteration_degree = sic - intervals[number - 1]

        if "d" in str(alteration_degree):
            alteration = -len([c for c in str(alteration_degree) if c.isalpha()])
        elif "a" in str(alteration_degree):
            alteration = len([c for c in str(alteration_degree) if c.isalpha()])
        elif "P" in str(alteration_degree):
            alteration = 0
        else:
            raise ValueError

        instance = cls(number=number, alteration=alteration)
        return instance

    @classmethod
    def from_fifth(cls, fifth: int, mode=Literal["major", "natural_minor"]) -> Self:

        sic = SpelledIntervalClass(fifth)

        instance = cls.from_sic(sic=sic, mode=mode)
        return instance

    def sic(self, mode: Literal["major", "minor"]) -> SpelledIntervalClass:
        """Return the spelled interval class from the reference tonic (scale degree 1)"""
        if self.alteration > 0:
            alteration_symbol = 'a' * self.alteration
        elif self.alteration == 0:
            alteration_symbol = 'P'  # perfect unison, no alterations
        elif self.alteration < 0:
            alteration_symbol = 'd' * abs(self.alteration)
        else:
            raise ValueError(self.alteration)

        if mode == 'major':
            intervals = intervals_in_key_dict['major']

        elif mode == 'natural_minor':
            intervals = intervals_in_key_dict['natural_minor']

        else:
            raise NotImplementedError(f'{mode=}')

        interval_alteration = SpelledIntervalClass(f'{alteration_symbol}1')
        interval_from_tonic = intervals[self.number - 1] + interval_alteration
        return interval_from_tonic

    def fifth(self, mode: Literal["major", "natural_minor"]) -> int:
        result = self.sic(mode=mode).fifths()
        return result


def test1():
    intervals = intervals_in_key_dict['major']
    C_scale_spcs = [SpelledPitchClass('C') + interval for interval in intervals]

    degree = Degree.from_string(degree_str='b6')

    if degree.alteration > 0:
        alteration_symbol = 'a' * degree.alteration
    elif degree.alteration == 0:
        alteration_symbol = 'P'  # perfect unison, no alterations
    elif degree.alteration < 0:
        alteration_symbol = 'd' * abs(degree.alteration)
    else:
        raise ValueError(degree.alteration)

    interval_alteration = SpelledIntervalClass(f'{alteration_symbol}1')
    print(f'{intervals=}')
    intermediate = intervals[degree.number - 1]
    print(f'{intermediate=}')
    interval_from_tonic_C = intervals[degree.number - 1] + interval_alteration
    print(f'{interval_alteration=}')
    print(f'{interval_from_tonic_C=}')
    print(type(interval_from_tonic_C))

    # result = SpelledIntervalClass(value=)
    #
    # print(f'{result=}')


def test():
    sic = SpelledIntervalClass('m3')
    print(f'{sic=}')
    result = Degree.from_sic(sic, mode="major")
    print(f'{result=}')


def test3():
    for i in range(-9, 9):
        dg = Degree.from_fifth(fifth=i, mode="natural_minor")
        print(f'{dg=}')
        result = dg.fifth(mode="natural_minor")
        print(f'{result=}')
        print("\n")


if __name__ == '__main__':
    test3()
