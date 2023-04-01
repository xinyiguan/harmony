import re
import typing
from abc import abstractmethod
from dataclasses import dataclass

from musana.harmony_types import SpelledPitchClass, Degree

@dataclass
class SpelledChord:
    root: SpelledPitchClass
    sds: typing.List[Degree]

    _chord_regex = re.compile("^(?P<class>[A-G])(?P<modifiers>(b*)|(#*))$")

    @classmethod
    def from_string(cls, string: str) -> typing.Self:
        root = ...
        sds = ...
        instance = cls(root=root, sds=sds)
        return instance