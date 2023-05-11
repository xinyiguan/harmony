from dataclasses import dataclass
from typing import Literal, Self

from pitchtypes import SpelledPitchClass, SpelledIntervalClass


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

    @classmethod
    def from_numeral_string(cls, numeral_string: str)->Self:


        root =...
        third=...
        fifth=...

        instance = cls(root=root, third=third,fifth=fifth)
        return  instance