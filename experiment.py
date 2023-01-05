# Created by Xinyi Guan in 2023.
import re
import typing
from dataclasses import dataclass
import regex_spm
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


@dataclass
class Quality:
    """Defined by the intervals between notes"""

    stacksize: int
    quality_list: typing.List[re.compile("^((M)|(m)|(a)+|(d)+)$")]

class SingleNumeralRegex:
    modifiers=re.compile("(b*)|(#*)?") # accidentals
    roman_numeral = re.compile("VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none")  # roman numeral
    form=re.compile("(%|o|\+|M|\+M)?" ) # form
    figbass=  re.compile("(7|65|43|42|2|64|6)?" ) # figured bass
    added_tones= re.compile("((\+)([#b])?([2-8]))+|([#b])?(9|1[0-4])")  # added tones, non-chord tones added within parentheses and preceded by a "+" or >8
    replacement_tones=re.compile("([#b])?([2-8])+")  # replaced chord tones expressed through intervals <= 8



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

    regex_matching_condition = lambda group_name: s_numeral_match[group_name] if s_numeral_match[group_name] else ''

    modifiers = regex_matching_condition('modifiers')
    roman_numeral = regex_matching_condition('roman_numeral')
    form = regex_matching_condition('form')
    figbass = regex_matching_condition('figbass')
    added_tones = regex_matching_condition('added_tones')
    replacement_tones = regex_matching_condition('replacement_tones')




    cond_M = roman_numeral.isupper() and figbass in ['', '6', '64']
    cond_m = ...
    cond_dim = ...
    cond_aug = ...

    cond_M7 = ...
    cond_7 = ...
    cond_m7 = ...
    cond_half_dim7 = ...
    cond_dim7 = ...
    cond_aug7 = ...

    quality_in_thirds_dict = {
        cond_M: ['M', 'm'], cond_m: ['m', 'M'], cond_dim: ['m', 'm'], cond_aug: ['M', 'M'],
        cond_M7: ['M', 'm', 'M'], cond_7: ['M', 'm', 'm'], cond_m7: ['m', 'M', 'm'],
        cond_half_dim7: ['m', 'm', 'M'], cond_dim7: ['m', 'm', 'm'], cond_aug7: ['M', 'M', 'd']
    }




def test_regex_spm():
    match regex_spm.fullmatch_in("123,45"):
        case r"(\d+),(?P<second>\d+)" as m:
            print("Notice the `as m` at the end of the line above")
            print(f"The first group is {m[1]}")
            print(f"The second group is {m['second']}")
            print(f"The full `re.Match` object is available as {m.match}")


if __name__ == '__main__':
    test_regex_spm()
