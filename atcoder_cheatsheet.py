# ruff: noqa: E402, F401  # import順を無視

import bisect
import heapq
import itertools
import math
import operator
import os
import sys
from collections import Counter, defaultdict, deque
from functools import cache, partial
from typing import Any, Callable, TypeVar

import numpy as np
from more_itertools import distinct_permutations  # 同じものを含む順列の数
from sortedcontainers import SortedDict, SortedList, SortedSet


def binary_search_max(func: Callable[[int], bool], bottom: int, top: int) -> int:
    """二分探索でfuncを満たす最大の値を探索する

    Args:
        func (callable[[int], bool]):
            探索対象の関数。引数に探索する値を取り、探索対象が真か偽かを返す。
        bottom (int): 探索範囲の下限
        top (int): 探索範囲の上限

    Returns:
        int: 探索した値

    Examples:
        >>> def func(x: int) -> bool:
        ...     return x <= 10
        >>> binary_search_max(func, 0, 100)
        10
    """
    top += 1  # topはfuncを満たさないことを前提とする
    # bottomはfuncを満たすことを前提とする

    while top - bottom > 1:
        mid = (top + bottom) // 2
        if func(mid):
            bottom = mid
        else:
            top = mid

    # top: ng
    # bottom: ok
    return bottom


def binary_search_min(func: Callable[[int], bool], bottom: int, top: int) -> int:
    """二分探索でfuncを満たす最小の値を探索する

    Args:
        func (callable[[int], bool]):
            探索対象の関数。引数に探索する値を取り、探索対象が真か偽かを返す。
        bottom (int): 探索範囲の下限
        top (int): 探索範囲の上限

    Returns:
        int: 探索した値

    Examples:
        >>> def func(x: int) -> bool:
        ...     return x >= 10
        >>> binary_search_min(func, 0, 100)
        10
    """
    # topはfuncを満たすことを前提とする
    bottom -= 1  # bottomはfuncを満たさないことを前提とする

    while top - bottom > 1:
        mid = (top + bottom) // 2
        if func(mid):
            top = mid
        else:
            bottom = mid

    # top: ok
    # bottom: ng
    return top


# 優先度付きキュー
class HeapQ:
    def __init__(self, list_: list[Any] | None = None) -> None:
        """優先度付きキューを初期化する

        Args:
            list_ (list[T], optional): 初期値. Defaults to [].
        """
        self._heap = list_ or []
        heapq.heapify(self._heap)

    def push(self, x: Any) -> None:
        """要素を追加する"""
        heapq.heappush(self._heap, x)

    def pop(self) -> Any:
        """最小値を取り出す"""
        return heapq.heappop(self._heap)

    def __bool__(self) -> bool:
        """要素があるかどうか"""
        return bool(self._heap)


class RollingHash():
    """ローリングハッシュ

    cf. https://kyoroid.github.io/algorithm/string/rolling_hash.html
    """

    def __init__(self, S, b=3491, m=999999937):
        """任意の基数と法でハッシュを生成する"""
        n = len(S)
        self.prefix = prefix = [0] * (n+1)
        self.power = power = [1] * (n+1)
        self.b = b
        self.m = m
        for i in range(n):
            c = ord(S[i])
            prefix[i+1] = (prefix[i] * b + c) % m
            power[i+1] = (power[i] * b) % m

    def get(self, l, r):
        """S[l, r) のハッシュを求める"""
        return (self.prefix[r] - self.power[r-l] * self.prefix[l]) % self.m

    def concat(self, h1, h2, l2):
        """S1+S2 のハッシュを、それぞれのハッシュから求める"""
        return (self.power[l2] * h1 + h2) % self.m

    def lcp(self, l1, r1, l2, r2):
        """S[l1, r1) とS[l2, r2) の最大共通接頭辞を求める"""
        # LCPの最小値 (範囲内)
        low = 0
        # LCPの最大値 + 1 (範囲外)
        high = min(r1-l1, r2-l2) + 1
        while high - low > 1:
            mid = (high + low) // 2
            if self.get(l1, l1 + mid) == self.get(l2, l2 + mid):
                low = mid
            else:
                high = mid
        return low

