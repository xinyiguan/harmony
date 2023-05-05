from typing import TypeVar, Callable, Optional, Dict, Any

import dimcat
import pandas as pd

from harmonytypes.numeral import Numeral

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def maybe_bind(f: Callable[[A], Optional[B]], g: Callable[[B], Optional[C]]) -> Callable[[A], Optional[C]]:
    def h(x: A) -> Optional[C]:
        y = f(x)
        if y is None:
            return None
        else:
            return g(y)

    return h

