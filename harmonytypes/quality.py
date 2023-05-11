import itertools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Self, List, Literal, Optional

from pitchtypes import SpelledIntervalClass, SpelledPitchClass, SpelledIntervalClassArray, asic, SpelledPitchClassArray, \
    aspc

from harmonytypes import theory
from harmonytypes.degree import Degree
from harmonytypes.basetypes import IntervalQuality, IP, P

ChordType = Literal["M", "m", "o", "+", "mm7", "Mm7", "MM7", "mM7", "o7", "%7", "+7", "+M7"]


# @dataclass
# class HarmonyQuality:
#     """ Examples:
#     stack_of_thirds_3_Mm = HarmonyQuality(stacksize=3,ic_quality_list=[IP(1),IP(-1)]) # Major triad
#     stack_of_thirds_4_mmm = HarmonyQuality(stacksize=3,ic_quality_list=[IP(-1),IP(-1),IP(-1)]) # fully diminished seventh chord
#     stack_of_fifth_4_PPP = HarmonyQuality(stacksize=5,ic_quality_list=[P(0),P(0),P(0)])
#     stack_of_fifth_4_ADP = HarmonyQuality(stacksize=5,ic_quality_list=[P(-1),P(1),P(0)])
#     """
#
#     stack_size: int
#     interval_class_quality: List[P | IP]
#
#     _thirds_map_to_ic_qualities_3 = {(True, '+'): [1, 1],  # "+": Augmented triad
#                                      (True, None): [1, -1],  # "M": Major triad
#                                      (False, None): [-1, 1],  # "m": Minor triad
#                                      (False, 'o'): [-1, -1]}  # "o": Diminished triad
#
#     _thirds_map_to_ic_qualities_4 = {(True, '+M'): [1, 1, -1],
#                                      # "+M7": augmented major 7th   =>  (M3, M3, m3), (1, 3, ♯5, 7)
#                                      (False, 'M'): [-1, 1, 1],
#                                      # "mM7": minor major 7th       =>  (m3, M3, M3), (1, ♭3, 5, 7)
#                                      (True, 'M'): [1, -1, 1],  # "MM7": major major 7th        =>  (M3 m3 M3), (1 3 5 7)
#                                      (True, '+'): [1, 1, -2],
#                                      # "+7": augmented minor 7th     =>  (M3, M3, d3), (1, 3, ♯5, ♭7)
#                                      (False, '%'): [-1, -1, 1],
#                                      # "%7": half dim 7th          =>  (m3 m3 M3), (1 ♭3 ♭5 ♭7)
#                                      (False, None): [-1, 1, -1],
#                                      # "mm7": minor minor 7th       =>  (m3 M3 m3), (1 ♭3 5 ♭7)
#                                      (True, None): [1, -1, -1],
#                                      # "Mm7": dominant 7th           =>  (M3 m3 m3), (1 3 5 ♭7)
#                                      (False, 'o'): [-1, -1,
#                                                     -1]}  # "o7": fully dim 7th        =>  (m3 m3 m3), (1 ♭3 ♭5 ♭♭7)
#
#     _which_dict = {3: _thirds_map_to_ic_qualities_3,
#                    4: _thirds_map_to_ic_qualities_4}
#
#     @classmethod
#     def parse_in_stack_of_thirds(cls,
#                                  numeral: Literal[
#                                      'VII', 'VI', 'V', 'IV', 'III', 'II', 'I', 'vii', 'vi', 'v', 'iv', 'iii', 'ii', 'i', 'Ger', 'It', 'Fr'],
#                                  form_symbol: Optional[Literal['o', '+', '%', 'M', '+M']],
#                                  figbass: Optional[Literal['7', '65', '43', '42', '2', '64', '6']]) -> Optional[Self]:
#
#         n_chord_tones = 3 if figbass in ['0', '6', '64'] else 4
#
#         dict_to_check = cls._which_dict[n_chord_tones]
#
#         if numeral in ['Ger', 'It', 'Fr']:
#             raise NotImplementedError
#         else:
#             upper_case = numeral.isupper()
#
#         print(f'{upper_case=}')
#         print(f'{form_symbol=}')
#
#         if (upper_case, form_symbol) not in dict_to_check:
#             raise ValueError()
#         ic_quality_list = [IP(x) for x in dict_to_check[(upper_case, form_symbol)]]
#
#         instance = cls(stack_size=3, interval_class_quality=ic_quality_list)
#         return instance
#
#     @property
#     def major_minor_mode(self) -> Literal['major', 'minor']:
#         mode = 'major' if (
#                 self.interval_class_quality[0] == IP(1) and self.interval_class_quality[
#             1] == IP(-1)) else 'minor'
#         return mode
#
#     def to_sic(self, root: SpelledPitchClass) -> List[SpelledIntervalClass]:
#         raise NotImplementedError

@dataclass
class TertianHarmonyQuality:
    quality_type: SpelledIntervalClassArray | Literal["Ger", "It", "Fr"]

    _ic_qualities_3 = {(True, '+'): [1, 1],  # "+": Augmented triad
                       (True, None): [1, -1],  # "M": Major triad
                       (False, None): [-1, 1],  # "m": Minor triad
                       (False, 'o'): [-1, -1]}  # "o": Diminished triad

    _ic_qualities_4 = {(True, '+M'): [1, 1, -1],  # "+M7": augmented major 7th   =>  (M3, M3, m3), (1, 3, ♯5, 7)
                       (False, 'M'): [-1, 1, 1],  # "mM7": minor major 7th       =>  (m3, M3, M3), (1, ♭3, 5, 7)
                       (True, 'M'): [1, -1, 1],  # "MM7": major major 7th       =>  (M3 m3 M3), (1 3 5 7)
                       (True, '+'): [1, 1, -2],  # "+7": augmented minor 7th    =>  (M3, M3, d3), (1, 3, ♯5, ♭7)
                       (False, '%'): [-1, -1, 1],  # "%7": half dim 7th           =>  (m3 m3 M3), (1 ♭3 ♭5 ♭7)
                       (False, None): [-1, 1, -1],  # "mm7": minor minor 7th       =>  (m3 M3 m3), (1 ♭3 5 ♭7)
                       (True, None): [1, -1, -1],  # "Mm7": dominant 7th          =>  (M3 m3 m3), (1 3 5 ♭7)
                       (False, 'o'): [-1, -1, -1]}  # "o7": fully dim 7th        =>  (m3 m3 m3), (1 ♭3 ♭5 ♭♭7)

    _which_dict = {3: _ic_qualities_3,
                   4: _ic_qualities_4}

    def __repr__(self):
        return f'{self.quality_type}'

    def __getitem__(self, idx):
        return self.quality_type[idx]

    @classmethod
    def parse(cls,
              numeral: Literal[
                  'VII', 'VI', 'V', 'IV', 'III', 'II', 'I', 'vii', 'vi', 'v', 'iv', 'iii', 'ii', 'i', 'Ger', 'It', 'Fr'],
              form_symbol: Optional[Literal['o', '+', '%', 'M', '+M']],
              figbass: Optional[Literal['7', '65', '43', '42', '2', '64', '6']]) -> Self:

        n_chord_tones = 3 if figbass in ['0', '6', '64'] else 4

        dict_to_check = cls._which_dict[n_chord_tones]

        if numeral in ['Ger', 'It', 'Fr']:  # TODO: handle aug 6th chords
            raise NotImplementedError
        else:
            upper_case = numeral.isupper()

        if (upper_case, form_symbol) not in dict_to_check:
            raise ValueError()
        interval_classes = asic([IP(x).to_interval_class(interval_number=3) for x in
                                 dict_to_check[(upper_case, form_symbol)]])

        instance = cls(quality_type=interval_classes)
        return instance

    @classmethod
    def from_chord_type_str(cls, chord_type: str) -> Optional[Self]:
        interval_classes = theory.chordtype_intervalclass_dict[chord_type]
        instance = cls(quality_type=interval_classes)
        return instance

    def to_aspc(self, tonic: SpelledPitchClass) -> SpelledPitchClassArray:
        accumulative_ic = list(itertools.accumulate(self.quality_type))
        sd_list = aspc([tonic] + [tonic + x for x in accumulative_ic])
        return sd_list

    def to_scale_degrees(self, tonic: SpelledPitchClass) -> List[Degree]:
        raise NotImplementedError

    def major_minor_mode(self) -> Literal["major", "natural_minor"]:
        if isinstance(self.quality_type, str):
            assert any(substring in self.quality_type for substring in ("Fr", "Ger", "It"))
            return "major"

        elif isinstance(self.quality_type, SpelledIntervalClassArray) and self.quality_type[0] == SpelledIntervalClass(
                "M3"):
            return "major"


        elif isinstance(self.quality_type, SpelledIntervalClassArray) and self.quality_type[0] == SpelledIntervalClass(
                "m3"):
            return "natural_minor"

        else:
            raise ValueError


def test():
    result = TertianHarmonyQuality.from_chord_type_str(chord_type='mM7')
    print(f'{result=}')


if __name__ == '__main__':
    test()
