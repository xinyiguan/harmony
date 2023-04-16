import re
from dataclasses import dataclass
from typing import List, Self

import regex_spm

from harmonytypes.degree import Degree

@dataclass
class FiguredBass:
    degrees: List[Degree]

    _figbass_degree_dict = {"7": [1, 3, 5, 7], "65": [3, 5, 7, 1], "43": [5, 7, 1, 3],
                            "42": [7, 1, 3, 5], "2": [7, 1, 3, 5],
                            "64": [5, 1, 3],
                            "6": [3, 5, 1]}  # assume the first number x is: root + x = bass , 1 := unison
    _figuredbass_regex = re.compile("(?P<figbass>(7|65|43|42|2|64|6))")

    @classmethod
    def parse(cls, figbass_str: str) -> Self:
        match = regex_spm.fullmatch_in(figbass_str)
        match match:
            case cls._figuredbass_regex:
                instance = cls(degrees=[Degree.from_string(degree_str=str(x))
                                        for x in FiguredBass._figbass_degree_dict.get(figbass_str)])
            case _:  # otherwise, assume root position triad
                instance = cls(degrees=[Degree(number=1, alteration=0),
                                        Degree(number=3, alteration=0),
                                        Degree(number=5, alteration=0)])
        return instance

    def n_chord_tones(self) -> int:
        return len(self.degrees)