import abc
from dataclasses import dataclass
from typing import Self

from pitchtypes import SpelledIntervalClass


class IntervalQuality:
    @classmethod
    @abc.abstractmethod
    def from_interval_class(cls, spelled_interval_class: SpelledIntervalClass) -> Self:
        pass


@dataclass
class P(IntervalQuality):
    """
    This is a type class for perfect intervals
    alt_steps = 0 means e.g. perfect unison, perfect fifths
    alt_steps = 1 means aug 1 (A1), aug 5(A5), aug 4...
    alt_steps = -1 means dim 1 (D1) ...
    """

    alt_steps: int

    @classmethod
    def from_interval_class(cls, spelled_interval_class: SpelledIntervalClass) -> Self:  # TODO: fill in
        pass


@dataclass
class IP(IntervalQuality):
    """
    This is a type class for imperfect intervals
    alt_steps = 1 means M2, M3, M6 ...
    alt_steps = 2 means a2, a3, a6 ...
    alt_steps = 3 means aa2, aa3, ...
    alt_steps = -1 means m2, m3, m6, ...
    alt_steps = -2 means d2, d3, d4, ....
    """
    alt_steps: int

    def __post_init__(self):
        if self.alt_steps == 0:
            raise ValueError(f'{self.alt_steps=}')
        self.alt_steps = self.alt_steps

    @classmethod
    def from_interval_class(cls, spelled_interval_class: SpelledIntervalClass) -> Self:  # TODO: fill in
        # check if input makes sense:

        alt_steps = ...

        instance = cls(alt_steps=alt_steps)
        return instance

    def to_interval_class(self, interval_number: int) -> SpelledIntervalClass:
        if self.alt_steps == 1:
            ic = SpelledIntervalClass(value=f'M{interval_number}')
        elif self.alt_steps > 1:
            alteration_symbol = 'a' * (interval_number - 1)
            ic = SpelledIntervalClass(value=f'{alteration_symbol}{interval_number}')
        elif self.alt_steps == -1:
            ic = SpelledIntervalClass(value=f'm{interval_number}')
        elif self.alt_steps < -1:
            alteration_symbol = 'd' * (interval_number - 1)
            ic = SpelledIntervalClass(value=f'{alteration_symbol}{interval_number}')
        else:
            raise ValueError
        return ic



class Harmony(abc.ABC):
    """
    The basic interface that is implemented by every chord and numeral (and pitch class) type.
    """

    # harmony operations

