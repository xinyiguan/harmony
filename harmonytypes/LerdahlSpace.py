from dataclasses import dataclass
from typing import List, Self, Dict

from pitchtypes import EnharmonicPitchClass, SpelledIntervalClassArray, SpelledPitchClass, aspc

from harmonytypes.chord import Triad
from harmonytypes.key import Key
from harmonytypes.numeral import Numeral
from harmonytypes.quality import TertianHarmonyQuality


@dataclass
class LerdahlDiatonicBasicSpace:
    a_root_space: List[EnharmonicPitchClass]
    b_fifth_space: List[EnharmonicPitchClass]
    c_chord_space: List[EnharmonicPitchClass]
    d_diatonic_space: List[EnharmonicPitchClass]
    e_chromatic_space: List[EnharmonicPitchClass]

    @classmethod
    def from_triad_with_key(cls, key: Key, triad: Triad) -> Self:
        a_root_space = [triad.root.convert_to(EnharmonicPitchClass)]
        b_fifth_space = [x.convert_to(EnharmonicPitchClass) for x in [triad.root, triad.fifth]]
        c_chord_space = [x.convert_to(EnharmonicPitchClass) for x in [triad.root, triad.third, triad.fifth]]
        d_diatonic_space = [x.convert_to(EnharmonicPitchClass) for x in key.get_scale()]
        e_chromatic_space = [EnharmonicPitchClass(i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        instance = cls(a_root_space=a_root_space,
                       b_fifth_space=b_fifth_space,
                       c_chord_space=c_chord_space,
                       d_diatonic_space=d_diatonic_space,
                       e_chromatic_space=e_chromatic_space)
        return instance

    @property
    def levels(self) -> Dict[str, List[EnharmonicPitchClass]]:
        result = {'a': self.a_root_space,
                  'b': self.b_fifth_space,
                  'c': self.c_chord_space,
                  'd': self.d_diatonic_space,
                  'e': self.e_chromatic_space
                  }
        return result

    @staticmethod
    def i(key1: Key, key2: Key) -> int:
        """
        regional_shifts_in_fifths
        equivalent to counting number of changes in flats in sharps in key signature
        """
        i = abs(key1.accidentals() - key2.accidentals())
        return i

    @staticmethod
    def j(chord1: Numeral, chord2: Numeral) -> int:
        """
        chordal_shifts_in_fifths
        """
        root1 = chord1.root
        root2 = chord2.root
        sic = root1.interval_to(root2)
        result = abs(sic.fifths())
        return result

    def k(self, target_basic_space: Self) -> int:
        """
        diff_in_basic_space: distinctive pcs
        """
        result = sum([len(set(level_target) - set(level_self))
                      for level_self, level_target in zip(self.levels.values(), target_basic_space.levels.values())])

        return result

    def chord_distance(self, chord1: Numeral, chord2: Numeral) -> int:
        distance = sum([self.i(key1=chord1.local_key, key2=chord2.local_key),
                        self.j(chord1=chord1, chord2=chord2),
                        self.k(target_basic_space=LerdahlDiatonicBasicSpace.from_triad_with_key(
                            key=chord2.local_key,
                            triad=Triad(root=chord2.root, third=chord2.third(), fifth=chord2.fifth())))
                        ])
        return distance


def test():
    chord1 = Numeral(numeral_string="I", bass=SpelledPitchClass("C"), root=SpelledPitchClass("C"),
                     spcs=aspc([SpelledPitchClass("C"), SpelledPitchClass("E"), SpelledPitchClass("G")]),
                     global_key=Key.from_string(key_str="C"), local_key=Key.from_string("C"),
                     harmony_quality=TertianHarmonyQuality(quality_type=SpelledIntervalClassArray(["M3", "m3"])))

    chord2 = Numeral(numeral_string="V", bass=SpelledPitchClass("G"), root=SpelledPitchClass("G"),
                     spcs=aspc([SpelledPitchClass("G"), SpelledPitchClass("B"), SpelledPitchClass("D")]),
                     global_key=Key.from_string(key_str="C"), local_key=Key.from_string("C"),
                     harmony_quality=TertianHarmonyQuality(quality_type=SpelledIntervalClassArray(["M3", "m3"])))


    triad = Triad(root=SpelledPitchClass("C"),
                  third=SpelledPitchClass("E"),
                  fifth=SpelledPitchClass("G"))

    triad2 = Triad(root=SpelledPitchClass("G"),
                   third=SpelledPitchClass("B"),
                   fifth=SpelledPitchClass("D"))

    space1 = LerdahlDiatonicBasicSpace.from_triad_with_key(key=Key.from_string("C"),
                                                           triad=triad)
    space2 = LerdahlDiatonicBasicSpace.from_triad_with_key(key=Key.from_string("C"),
                                                           triad=triad2)

    # result = space1.k(target_basic_space=space2)
    # result = space1.j(chord1, chord2)
    result = space1.chord_distance(chord1, chord2)

    print(type(result))
    print(f'{result=}')


def test2():
    set1 = [SpelledPitchClass("C"), SpelledPitchClass("C"), SpelledPitchClass("G")]
    set2 = [SpelledPitchClass("C"), SpelledPitchClass("A"), SpelledPitchClass("B")]
    result = set2 - set1
    print(f'{result=}')


if __name__ == "__main__":
    test()
