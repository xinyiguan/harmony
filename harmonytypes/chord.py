from dataclasses import dataclass
from typing import Self, List, Literal, Optional
from pitchtypes import SpelledPitchClassArray, SpelledPitchClass, aspc, SpelledIntervalClass

import pandas as pd

from harmonytypes.key import Key
from harmonytypes.quality import TertianHarmonyQuality


@dataclass
class Triad:
    root: SpelledPitchClass
    third: SpelledPitchClass
    fifth: SpelledPitchClass

    @classmethod
    def from_root_quality(cls, root: SpelledPitchClass, quality: Literal["major", "minor"]) -> Self:
        if quality == "major":
            third = root + SpelledIntervalClass("M3")
        elif "minor" in quality:
            third = root + SpelledIntervalClass("m3")
        else:
            raise ValueError
        fifth = root + SpelledIntervalClass("P5")
        instance = cls(root=root,
                       third=third,
                       fifth=fifth)
        return instance


@dataclass
class SpelledChord:
    pcs: SpelledPitchClassArray
    root: Optional[SpelledPitchClass]
    bass: Optional[SpelledPitchClass]
    types: TertianHarmonyQuality

    @classmethod
    def from_string(cls, string: str) -> Self:
        instance = cls()
        return instance

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        all_pcs = df['chord_tones'] + df['added_tones']
        instance = cls()
        return instance

    def non_diatonic_pcs(self, key: Key) -> List[SpelledPitchClass]:
        ndpcs = [x for x in self.pcs if x not in key.get_scale()]
        return ndpcs


def test():
    chord = Chord(pcs=aspc(things=[0, 7, 1]))
    print(f'{chord=}')
    result = chord.non_diatonic_pcs(key=Key.from_string('C'))
    print(f'{result=}')


if __name__ == '__main__':
    test()
