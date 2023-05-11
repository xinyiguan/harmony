# defining concepts in Schenkerian theory
import re
from dataclasses import dataclass
from typing import Literal, List, TypeVar, Generic, Self

import pandas as pd
from pitchtypes import SpelledPitchClass

# HARD ENCODING =====================================================================================================
# dictionary mapping the stufen str(with its harmonic implication) in a system to its relevant position (int) to tonic

major_diatonic_system_dict = {"I": 1, "ii": 2, "iii": 3, "IV": 4, "V": 5, "vi": 6, "viio": 7}

minor_diatonic_system_dict = {"i": 1, "bII": 2, "iio": 2, "bIII": 3, "iv": 4, "v": 5, "bVI": 6, "bVII": 7}

major_secondary_system_dict = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "vii": 7, "VII": 7}

minor_secondary_system_dict = {"i": 1, "bii": 2, "iio": 2, "biii": 3, "iv": 4, "v": 5, "bvi": 6, "bvii": 7}

# The combined major-minor system
combined_major_minor_system_dict = {"I": 1, "i": 1,
                                    "ii": 2, "bII": 2, "iio": 2, "II": 2, "bii": 2,
                                    "iii": 3, "bIII": 3, "III": 3, "biii": 3,
                                    "IV": 4, "iv": 4,
                                    "V": 5, "v": 5,
                                    "vi": 6, "bVI": 6, "VI": 6, "bvi": 6,
                                    "viio": 7, "bVII": 7, "VII": 7, "bvii": 7}


simple_mixture_dict = {"I": ["i"], "ii": ["bII", "iio"], "iii": ["bIII"], "IV": ["iv"],
                       "V": ["v"], "vi": ["bVI"], "viio": ["bVII"],

                       "i": ["I"], "bII": ["ii"], "iio": ["ii"], "bIII": ["iii"], "iv": ["IV"],
                       "v": ["V"], "bVI": ["vi"], "bVII": ["viio"]}

secondary_mixture_dict = {"I": ["I"], "ii": ["II"], "iii": ["III"], "IV": ["IV"],
                          "V": ["V"], "vi": ["VI"], "viio": ["vii", "VII"],

                          "i": ["i"], "bII": ["bii"], "iio": ["iio"], "bIII": ["biii"], "iv": ["iv"],
                          "v": ["v"], "bVI": ["bvi"], "bVII": ["bvii"]}

double_mixture_dict = {"I": ["i"], "ii": ["bii", "iio"], "iii": ["biii"], "IV": ["iv"],
                       "V": ["v"], "vi": ["bvi"], "viio": ["bvii"],

                       "i": ["I"], "bII": ["II"], "iio": ["II"], "bIII": ["III"], "iv": ["IV"],
                       "v": ["V"], "bVI": ["VI"], "bVII": ["vii", "VII"]}


@dataclass
class Stufe:
    stufen_str: str

    @classmethod
    def from_string(cls, stufen_str: str) -> Self:
        stufen_regex = re.compile("^(?P<alteration>(b*)|(#*))"
                                  "(?P<numeral>(I|i|II|ii|III|iii|IV|iv|V|v|VI|vi|VII|vii))"
                                  "(?P<diminshed>(o*))$")

        # check input string regex:
        if not isinstance(stufen_str, str):
            raise TypeError(f"expected string as input, got {stufen_str}")
        match = stufen_regex.fullmatch(stufen_str)
        if match is None:
            raise ValueError(f"could not match '{stufen_str}' with regex: '{stufen_regex.pattern}'")

        instance = cls(stufen_str=stufen_str)
        return instance

    def simple_mixture(self) -> List[Self]:
        """Transform the current stufe to another via simple mixture"""
        if self.stufen_str in major_diatonic_system_dict:
            value = major_diatonic_system_dict.get(self.stufen_str)
            resulting_stufen = [Stufe.from_string(k) for k, v in minor_diatonic_system_dict.items() if v == value]

        elif self.stufen_str in minor_diatonic_system_dict:
            value = minor_diatonic_system_dict.get(self.stufen_str)
            resulting_stufen = [Stufe.from_string(k) for k, v in major_diatonic_system_dict.items() if v == value]

        else:
            raise ValueError

        return resulting_stufen

    def secondary_mixture(self) -> List[Self]:
        if self.stufen_str in major_diatonic_system_dict:
            value = major_secondary_system_dict.get(self.stufen_str)
            resulting_stufen = [Stufe.from_string(k) for k, v in minor_secondary_system_dict.items() if v == value]

        elif self.stufen_str in minor_diatonic_system_dict:
            value = minor_secondary_system_dict.get(self.stufen_str)
            resulting_stufen = [Stufe.from_string(k) for k, v in major_secondary_system_dict.items() if v == value]

        else:
            raise ValueError

        return resulting_stufen

    def double_mixture(self) -> Self:
        if self.stufen_str in major_diatonic_system_dict:
            value = major_secondary_system_dict.get(self.stufen_str)
            resulting_stufen = [Stufe.from_string(k) for k, v in minor_secondary_system_dict.items() if v == value]

        elif self.stufen_str in minor_diatonic_system_dict:
            value = minor_secondary_system_dict.get(self.stufen_str)
            resulting_stufen = [Stufe.from_string(k) for k, v in major_secondary_system_dict.items() if v == value]

        else:
            raise ValueError

        return resulting_stufen
