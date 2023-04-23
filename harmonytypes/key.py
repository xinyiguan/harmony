from __future__ import annotations

import re
from typing import Literal, Self, List
from dataclasses import dataclass
import numpy as np
from pitchtypes import SpelledPitchClass, SpelledPitchClassArray, asic, aspc, SpelledIntervalClass

from harmonytypes.degree import Degree
from harmonytypes.theory import intervals_in_key_dict


@dataclass(frozen=True)
class Key:
    tonic: SpelledPitchClass
    mode: Literal['major', 'natural_minor', 'melodic_minor', 'harmonic_minor',
    'ionian', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']

    key_interval_dict = {'major': asic(things=np.array(['P1', 'M2', 'M3', 'P4', 'P5', 'M6', 'M7'])),
                         'natural_minor': asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'm7'])),
                         'harmonic_minor': asic(things=np.array(['P1', 'M2', 'm3', 'P4', 'P5', 'm6', 'M7']))
                         }

    def __repr__(self):
        if self.mode == 'major':
            shown = f'{self.tonic}'
        elif self.mode == 'natural_minor':
            shown = f'{self.tonic}'.lower()
        else:
            raise NotImplementedError
        return f'{shown}'

    @classmethod
    def from_string(cls, key_str: str) -> Self:
        _key_regex = re.compile("^(?P<class>[A-G])(?P<modifiers>(b*)|(#*))$", re.I)

        if not isinstance(key_str, str):
            raise TypeError(f"Expected string as input, got {key_str}")
        key_match = _key_regex.fullmatch(key_str)
        if key_match is None:
            raise ValueError(f"Could not match '{key_str}' with regex: '{_key_regex.pattern}'")

        mode = 'major' if key_match['class'].isupper() else 'natural_minor'
        tonic = SpelledPitchClass(key_match['class'].upper() + key_match['modifiers'])
        instance = cls(tonic=tonic, mode=mode)
        return instance

    @classmethod
    def get_local_key(cls, global_key: Self, rn_degree: str) -> Self:
        localkey_mode = 'major' if rn_degree.isupper() else 'natural_minor'
        localkey_tonic = global_key.find_spc(degree=Degree.from_string(degree_str=rn_degree))
        instance = cls(tonic=localkey_tonic,
                       mode=localkey_mode)
        return instance

    def parallel(self) -> Key:
        raise NotImplementedError

    def relative(self) -> Key:
        if self.mode == 'major':
            new_tonic = self.tonic - SpelledIntervalClass('m3')
            relative_minor = Key(tonic=new_tonic, mode='natural_minor')
            return relative_minor
        else:
            new_tonic = self.tonic + SpelledIntervalClass('m3')
            relative_major = Key(tonic=new_tonic, mode='major')
            return relative_major

    def accidentals(self) -> int:
        return abs(self.relative().tonic.fifths()) if self.mode == 'minor' else abs(self.tonic.fifths())

    def get_scale(self) -> List[SpelledPitchClass]:
        intervals = self.key_interval_dict[self.mode]
        scale = [self.tonic + intervals[i] for i in range(len(intervals))]
        return scale

    @staticmethod
    def find_spc(key: Key, degree: Degree) -> SpelledPitchClass:
        if key.mode == 'major':
            intervals = key.key_interval_dict['major']
        elif key.mode == 'natural_minor':
            intervals = key.key_interval_dict['natural_minor']
        else:
            raise NotImplementedError(f'{key.mode=}')

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
        spc = key.tonic + interval
        return spc

    def find_degree(self, spc: SpelledPitchClass) -> Degree:
        """
        Example: in Db major scale,
                - F# will be #3
                - E will be #2
                - E# will be ##2
        """
        scale_without_accidentals = [x.letter() for x in self.get_scale()]
        position_in_scale = scale_without_accidentals.index(spc.letter())
        degree_num_part = position_in_scale + 1

        target_alteration = spc.alteration()
        original_scale_alteration = self.get_scale()[position_in_scale].alteration()
        alteration_num = target_alteration - original_scale_alteration

        if alteration_num > 0:
            alteration_symbol = '#' * alteration_num
        elif alteration_num == 0:
            alteration_symbol = ''
        elif alteration_num < 0:
            alteration_symbol = 'b' * abs(alteration_num)
        else:
            raise ValueError(spc.alteration)

        ensemble_degree_string = alteration_symbol + str(degree_num_part)

        result_degree = Degree.from_string(degree_str=ensemble_degree_string)
        return result_degree

    def to_string(self) -> str:
        return self.__repr__()


def test():
    key = Key.from_string(key_str='Db')
    result = key.find_degree(spc=SpelledPitchClass("Bbb"))
    print(f'{result=}')


if __name__ == '__main__':
    test()
