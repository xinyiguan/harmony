from abc import ABCMeta
from dataclasses import dataclass

import numpy as np
import pitchtypes
from pitchtypes import SpelledIntervalClass, SpelledPitchClass, EnharmonicPitchClass

from musana.harmony_types import Key, Degree, TonalHarmony, Triad
import typing



@dataclass
class LerdahlDiatonicBasicSpace:
    a_root_space: typing.List[EnharmonicPitchClass]
    b_fifth_space: typing.List[EnharmonicPitchClass]
    c_chord_space: typing.List[EnharmonicPitchClass]
    d_diatonic_space: typing.List[EnharmonicPitchClass]
    e_chromatic_space: typing.List[EnharmonicPitchClass]

    @classmethod
    def from_triad_with_key(cls, key: Key, triad: Triad) -> typing.Self:
        a_root_space = [EnharmonicPitchClass(triad.root)]
        b_fifth_space = [EnharmonicPitchClass(x) for x in [triad.root, triad.fifth]]
        c_chord_space = [EnharmonicPitchClass(x) for x in [triad.root, triad.third, triad.fifth]]
        d_diatonic_space = [EnharmonicPitchClass(x) for x in key.scale()]
        e_chromatic_space = [EnharmonicPitchClass(i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        instance = cls(a_root_space=a_root_space,
                       b_fifth_space=b_fifth_space,
                       c_chord_space=c_chord_space,
                       d_diatonic_space=d_diatonic_space,
                       e_chromatic_space=e_chromatic_space)
        return instance

    @property
    def levels(self) -> typing.Dict[str, typing.List[EnharmonicPitchClass]]:
        result = {'a': self.a_root_space,
                  'b': self.b_fifth_space,
                  'c': self.c_chord_space,
                  'd': self.d_diatonic_space,
                  'e': self.e_chromatic_space
                  }
        return result

    # @classmethod
    # def _from_root(cls, root: EnharmonicPitchClass):
    #     levels = {}
    #     for level, intervals in LerdahlDiatonicBasicSpace._pc_level_dict.items():
    #         levels[level] = [root + EnharmonicIntervalClass(i) for i in intervals]
    #
    #     instance = cls(**levels)
    #     return instance

    @staticmethod
    def i(key1: Key, key2: Key) -> int:
        """
        regional_shifts_in_fifths
        equivalent to counting number of changes in flats in sharps in key signature
        """
        i = abs(key1.accidentals - key2.accidentals)
        return i

    @staticmethod
    def j(chord1: TonalHarmony, chord2: TonalHarmony) -> int:
        """
        chordal_shifts_in_fifths
        """
        root1 = chord1.root()
        root2 =chord2.root()
        sic = root1.interval_to(root2)
        result = abs(sic.fifths())
        return result

    def k(self, target_basic_space: typing.Self) -> int:
        """
        diff_in_basic_space: distinctive pcs
        """
        result = sum([set(level_target) - (set(level_self))
                      for level_self, level_target in zip(self.levels.values(), target_basic_space.levels.values())])
        return result

    def chord_distance(self, chord1: TonalHarmony, chord2: TonalHarmony) -> int:
        distance = sum([self.i(key1=chord1.localkey, key2=chord2.localkey),
                        self.j(chord1=chord1, chord2=chord2),
                        self.k(target_basic_space=LerdahlDiatonicBasicSpace.from_triad_with_key(
                            key=chord2.localkey,
                            triad=Triad(root=chord2.root(), third=chord2.third(), fifth=chord2.fifth())))
                        ])
        return distance


class PitchClassLevel(LerdahlDiatonicBasicSpace):

    def vertical_distance_from_pc0(self):
        raise NotImplementedError

    def horizontal_distance_from_pc0(self):
        raise NotImplementedError

    def combined_distance_from_pc0(self):
        raise NotImplementedError


def test():
    pass

if __name__ == '__main__':
    test()