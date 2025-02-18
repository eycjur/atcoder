# ruff: noqa: E402, F401  # import順を無視


import bisect
import heapq
import math
import operator
import os
import random
import sys
from collections import Counter, defaultdict, deque
from functools import cache, cmp_to_key, partial
from itertools import (
    accumulate,  # 累積値
    chain,  # 平坦化
    combinations,  # 順列
    combinations_with_replacement,  # 重複順列
    permutations,  # 組み合わせ
    product,  # 直積（重複組み合わせ）
)
from math import (
    ceil,  # 切り上げ
    comb,  # 組み合わせ(nCk)
    factorial,  # 階乗
    floor,  # 切り捨て
    gcd,  # 最大公約数
    inf,
    isqrt,  # 平方根
    lcm,  # 最小公倍数
    log,
    log10,  # 常用対数(log(x, 10)より高精度)
    perm,  # 順列(nPk)
    pow,  # 冪乗
    sqrt,  # 平方根
)
from string import ascii_letters, ascii_lowercase, ascii_uppercase
from typing import Any, Callable, TypeVar

# import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from atcoder.dsu import DSU as UnionFind  # 連結判定
from atcoder.fenwicktree import FenwickTree  # 部分和
from atcoder.lazysegtree import LazySegTree  # 区間更新および区間クエリ
from atcoder.math import inv_mod  # 逆元
from atcoder.segtree import SegTree  # 区間クエリ
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


class HeapQ:
    """優先度付きキュー

    Args:
        list_ (list[T], optional): 初期値. Defaults to [].

    Attributes:
        _heap (list[T]): ヒープ

    Methods:
        push(x: T) -> None:
            要素を追加する
        pop() -> T:
            最小値を取り出す
        __bool__() -> bool:
            要素があるかどうか

    Examples:
        >>> q = HeapQ([3, 1, 4, 9, 5])
        >>> q
        [1, 3, 4, 9, 5]
        >>> q.push(7)
        >>> q.pop()
        1
        >>> q
        [3, 5, 4, 9, 7]
    """
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

    def __str__(self) -> str:
        return str(sorted(self._heap))


# ローリングハッシュ
class RollingHash():
    """ローリングハッシュ

    Example:
        >>> lcp = 10**9
        >>> for m in [999999937, 10**9+7]:
        ...     b = random.randint(2, m-1)
        ...     rh = RollingHash("monoraimonoid", m=m, b=b)
        ...     lcp = min(lcp, rh.lcp(0, 7, 7, 13))
        >>> print(lcp)
        4  # "mono"

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


# グラフの可視化
def plot_graph_by_dict(data: dict | list, is_node_start_one: bool = True, is_directed: bool = False) -> None:
    """グラフを可視化して、graph.pngに保存する

    Args:
        data (dict | list): グラフのデータ
            # weighted
                list: [[(node_to1, weight1), (node_to2, weight2)], [], ...]
                dict: {node_from1: [(node_to1, weight1), (node_to2, weight2)], node_from2: [], ...}
            # not weighted
                list: [[node_to1, node_to2], [], ...]
                dict: {node_from1: [node_to1, node_to2], node_from2: [], ...}
        is_node_start_one (bool, optional): ノード番号を1から始めるか. Defaults to True.
        is_directed (bool, optional): 有向グラフか. Defaults to False
    """
    if isinstance(data, dict):
        iter_node_edges = data.items()
    else:
        iter_node_edges = enumerate(data)

    if is_directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    node_offset = 1 if is_node_start_one else 0
    for i in range(len(data)):
        if is_node_start_one:
            graph.add_node(i + node_offset)
        else:
            graph.add_node(i)

    for node, edges in iter_node_edges:
        for edge in edges:
            is_weighted = isinstance(edge, tuple) or isinstance(edge, list)
            if is_weighted:
                graph.add_edge(node + node_offset, edge[0] + node_offset, weight=edge[1])
            else:
                graph.add_edge(node + node_offset, edge + node_offset)
    pos = nx.spring_layout(graph)  # 見づらい場合は複数回繰り返すかspiral_layoutなどを検討
    nx.draw(graph, pos=pos, with_labels=True, arrows=is_directed)
    if is_weighted:
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        print(edge_labels)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.savefig("graph.png")
