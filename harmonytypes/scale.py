from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, List, Literal, Set

from pitchtypes import asic, SpelledPitchClass, aspc, EnharmonicPitchClass, SpelledPitchArray, EnharmonicIntervalClass


@dataclass
class AbstractScale(ABC):
    tonic: SpelledPitchClass | EnharmonicPitchClass
    members: aspc | List[SpelledPitchClass] | List[EnharmonicPitchClass]
    name: str

    @classmethod
    @abstractmethod
    def from_tonic(cls, tonic: SpelledPitchClass | EnharmonicPitchClass, mode: str) -> Self:
        raise NotImplementedError

    @abstractmethod
    def check_membership(self, pitch: EnharmonicPitchClass | SpelledPitchClass) -> bool:
        raise NotImplementedError


@dataclass
class DiatonicScale(AbstractScale):
    tonic: SpelledPitchClass
    members: List[SpelledPitchClass]
    name: str

    _major_diatonic_interval_sequence = asic(things=["P1", "M2", "M3", "P4", "P5", "M6", "M7"])
    _minor_diatonic_interval_sequence = asic(things=["P1", "M2", "m3", "P4", "P5", "m6", "m7"])

    @classmethod
    def from_tonic(cls, tonic: SpelledPitchClass, mode: Literal["major", "minor"]) -> Self:
        if mode == "major":
            intervals_sequence = DiatonicScale._major_diatonic_interval_sequence
        elif mode == "minor":
            intervals_sequence = DiatonicScale._minor_diatonic_interval_sequence
        else:
            raise ValueError

        members = aspc([tonic + x for x in intervals_sequence])
        name = f"{tonic} {mode} diatonic"

        instance = cls(tonic=tonic, members=members, name=name)
        return instance

    def check_membership(self, pitch: SpelledPitchClass) -> bool:
        result = pitch in self.members
        return result


@dataclass
class OctatonicScale(AbstractScale):
    tonic: EnharmonicPitchClass
    members: List[EnharmonicPitchClass]
    name: str

    _whole_half_interval_sequence = [EnharmonicIntervalClass(x) for x in
                                     ["P1", "M2", "m3", "P4", "a4", "a5", "M6", "M7"]]
    _half_whole_interval_sequence = [EnharmonicIntervalClass(x) for x in
                                     ["P1", "m2", "m3", "d4", "a4", "P5", "M6", "m7"]]

    @classmethod
    def from_tonic(cls, tonic: EnharmonicPitchClass, mode: Literal["whole-half", "half-whole"]) -> Self:
        if mode == "whole-half":
            intervals_sequence = OctatonicScale._whole_half_interval_sequence
        elif mode == "half-whole":
            intervals_sequence = OctatonicScale._half_whole_interval_sequence
        else:
            raise ValueError

        members = [tonic + x for x in intervals_sequence]
        name = f"{tonic} {mode}"

        instance = cls(tonic=tonic, members=members, name=name)
        return instance

    def check_membership(self, pitch: EnharmonicPitchClass) -> bool:
        result = pitch in self.members
        return result


@dataclass
class HexatonicScale(AbstractScale):
    tonic: EnharmonicPitchClass
    members: List[EnharmonicPitchClass]
    name: str

    _13_hexatonic_interval_sequence = [EnharmonicIntervalClass(x) for x in
                                          ["P1", "m2", "M3", "P4", "a5", "M6"]]
    _31_hexatonic_interval_sequence = [EnharmonicIntervalClass(x) for x in
                                          ["P1", "m3", "M3", "P5", "m6", "M7"]]

    @classmethod
    def from_tonic(cls, tonic: EnharmonicPitchClass, mode: Literal["13", "31"]) -> Self:
        if mode == "13":
            intervals_sequence = HexatonicScale._13_hexatonic_interval_sequence
        elif mode == "31":
            intervals_sequence = HexatonicScale._31_hexatonic_interval_sequence
        else:
            raise ValueError

        members = [tonic + x for x in intervals_sequence]
        name = f"{tonic} {mode} hexatonic"

        instance = cls(tonic=tonic, members=members, name=name)
        return instance

    def check_membership(self, pitch: EnharmonicPitchClass) -> bool:
        result = pitch in self.members
        return result


if __name__ == "__main__":
    oc = HexatonicScale.from_tonic(tonic=EnharmonicPitchClass("C"), mode="13")
    print(f'{oc=}')

    oeb = HexatonicScale.from_tonic(tonic=EnharmonicPitchClass("E"), mode="13")
    print(f'{oeb=}')

    oa = HexatonicScale.from_tonic(tonic=EnharmonicPitchClass("Ab"), mode="13")
    print(f'{oa=}')

    print(set(oc.members) == set(oeb.members) == set(oa.members))
