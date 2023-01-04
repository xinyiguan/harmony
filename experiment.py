# Created by Xinyi Guan in 2023.
import re
import typing
from dataclasses import dataclass

from pitchtypes import SpelledPitchClass, SpelledIntervalClass


@dataclass
class Degree:
    number: int
    alteration: int | bool  # when int: positive for "#", negative for "b", when bool: represent whether to use natural

    def __add__(self, other: typing.Self) -> typing.Self:
        number = (self.number + other.number - 1) % 7
        alteration = other.alteration
        return Degree(number=number, alteration=alteration)

    @classmethod
    def parse(cls, scale_degree: str) -> typing.Self:
        sd_regex = re.compile("^((?P<modifiers>(b*)|(#*))?(?P<number>([0-9]+)))$")
        sd_match = sd_regex.match(scale_degree)
        if sd_match is None:
            raise ValueError(f"could not match '{scale_degree}' with regex: '{sd_regex.pattern}'")

        number = sd_match['number']
        modifiers = sd_match['modifiers']
        alteration = len(modifiers) if '#' in modifiers else -len(modifiers)

        # create class instance:
        instance = cls(number=number, alteration=alteration)
        return instance


def test_parse():
    result = Degree.parse(scale_degree='##2')
    print(result)


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

    s_numeral_match = _sn_regex.match("#viio65(4)")
    print(s_numeral_match)
    modifiers = s_numeral_match['modifiers']
    form = s_numeral_match['form']
    figbass = s_numeral_match['figbass']
    added_tones_match = s_numeral_match['added_tones']
    replacement_tones_match = s_numeral_match['replacement_tones']
    print('modifiers: ', modifiers)
    print('form: ', form)
    print('figbass: ', figbass)
    print('added_tones:  ', added_tones_match)
    print('replacement_tones: ', replacement_tones_match)

    if added_tones_match:
        added_tones_degrees = [Degree.parse(scale_degree=x) for x in added_tones_match.split('+')[1:]]
        print('added_tones_degrees: ', added_tones_degrees)

    if replacement_tones_match:
        seperated_replaced_tones_tuples = re.findall(r'([#b])?([2-8])', string=replacement_tones_match)
        seperated_replaced_tones = [''.join(x) for x in seperated_replaced_tones_tuples]
        replacement_tones_degrees = [Degree.parse(scale_degree=x) for x in seperated_replaced_tones]
        print('replacement_tones_degrees: ', replacement_tones_degrees)

    figbass_match = s_numeral_match['figbass']
    if figbass_match:
        figbass_degree_dict = {"7": [1, 3, 5, 7], "65": [1, 3, 5, 6], "43": [1, 3, 4, 6],
                               "42": [1, 2, 4, 6], "2": [1, 2, 4, 6],
                               "64": [1, 4, 6], "6": [1, 3, 6]}  # TODO: need to double check
        figbass_list = list(map(str,figbass_degree_dict.get(figbass_match)))
        figbass = list(map(Degree.parse, figbass_list))

        print('figbass: ', figbass)



if __name__ == '__main__':
    test()
