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


MOD_998 = 998244353
MOD_1007 = 10 ** 9 + 7

def binary_search(func: Callable[[int], bool], ok: int, ng: int) -> int:
    """二分探索でfuncを満たす最大/最小の値を探索する

    Args:
        func (callable[[int], bool]):
            探索対象の関数。引数に探索する値を取り、探索対象が真か偽かを返す。
        ok (int): 条件を満たす値
        top (int): 条件を満たさない値

    Returns:
        int: 条件を満たす最大/最小の値

    Examples:
        >>> def func(x: int) -> bool:
        ...     return x >= 10
        >>> binary_search(func, 0, 100)
        10
    """
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok


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
class RollingHash:
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


class CustomHash:
    """カスタムハッシュクラス

    文字列のハッシュ値を計算するクラス

    \_hash(S) = sum_{1 <= i <= n} (ord(S[i]) * base^(n - i)) % mod
    hash(S) = (_hash(S), _hash(S)
    S: 文字列
    n: 文字列の長さ

    Args:
        bases (tuple[int, int], optional): ハッシュの基数. Defaults to (257, 1009).
        mods (tuple[int, int], optional): ハッシュの法. Defaults to (999999937, 10**9 + 7).

    Methods:
        hash(string: str) -> tuple[int, int]:
            文字列のハッシュ値を計算する
        hash_each(string: str) -> list[tuple[int, int]]:
            文字列の各文字のハッシュ値を計算する

    Examples:
        >>> ch = CustomHash()
        >>> ch.hash("abc")
        (6432038, 98852838)
        >>> ch.hash_each("abc")
        [(97, 97), (25027, 97971), (6432038, 98852838)]
    """

    def __init__(self, bases: tuple[int, int] = (257, 1009), mods: tuple[int, int] = (999999937, 10**9 + 7)) -> None:
        self.bases = bases
        self.mods = mods

    def _hash(self, string: str, base: int, mod: int) -> int:
        ret = 0
        for s in string:
            ret = (ret * base + ord(s)) % mod
        return ret

    def _hash_from_prev(self, prev_hash: tuple[int, int], new_char: str) -> tuple[int, int]:
        return tuple(
            (prev * base + ord(new_char)) % mod
            for prev, base, mod in zip(prev_hash, self.bases, self.mods)
        )

    def hash(self, string: str) -> tuple[int, int]:
        """文字列のハッシュ値を計算する

        Args:
            string (str): ハッシュ化する文字列

        Returns:
            tuple[int, int]: ハッシュ値
        """
        return tuple(
            self._hash(string, base, mod)
            for base, mod in zip(self.bases, self.mods)
        )

    def hash_each(self, string: str) -> list[tuple[int, int]]:
        """文字列の各文字のハッシュ値を計算する

        Args:
            string (str): ハッシュ化する文字列
        Returns:
            list[tuple[int, int]]: ハッシュ値のリスト
        """
        hash_list = [self.hash(string[0])]
        for i in range(1, len(string)):
            hash_list.append(self._hash_from_prev(hash_list[i - 1], string[i]))
        return hash_list


class Trie:
    """Trie木の実装

    接頭辞を利用した検索を行うためのデータ構造。

    Example:
        >>> trie = Trie()
        >>> trie.insert("fire")
        0
        >>> trie.insert("fish")
        2  # これまでの単語との共通接頭辞の長さの和
        >>> trie
        root (2)
            f (2)
                i (2)
                    r (1)
                        e (1)★
                    s (1)
                        h (1)★
        >>> trie.num_start_with("fire")
        1
        >>> trie.start_with("hoge")
        False
        >>> trie.count()  # 挿入した単語数
        2
    """
    class Node:
        def __init__(self) -> None:
            self.children = {}
            self.count = 0  # 何単語がこのノードを経由したか

    def __init__(self):
        """Trie木を初期化する"""
        self.root = self.Node()

    def insert(self, word: str) -> int:
        """Trie木に単語を挿入する

        Args:
            word: 挿入する単語

        Returns:
            ans: これまでの単語との共通接頭辞の長さの和
        """
        node = self.root
        self.root.count += 1
        sum_prefix_length = 0
        for char in word:
            if char not in node.children:
                # 子ノードがなければ新規追加
                node.children[char] = self.Node()

            node = node.children[char]
            sum_prefix_length += node.count
            # このノードを通過する単語数をカウント
            node.count += 1
        return sum_prefix_length

    def num_start_with(self, prefix: str) -> bool:
        """prefix で始まる単語の数を返す

        Args:
            prefix: 接頭辞
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.count

    def start_with(self, prefix: str) -> bool:
        """prefix で始まる単語が存在するかを返す

        Args:
            prefix: 検索する単語
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def count(self) -> int:
        """挿入済み単語数を返す"""
        return self.root.count

    def __str__(self):
        """Trie の構造を文字列で返す"""
        def dfs(node: Trie.Node, depth: int, char: str) -> str:
            str_ = "    " * depth
            if char == "":
                str_ += f"root ({node.count})"
            else:
                str_ += f"{char} ({node.count})"
                if node.count > sum([child.count for child in node.children.values()]):
                    str_ += "★"
            for next_char, next_node in node.children.items():
                str_ += f"\n{dfs(next_node, depth + 1, next_char)}"
            return str_

        return dfs(self.root, 0, "")


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


def pop_count(n: int, k: int, mod: int | None = None) -> tuple[int, int]:
    """n桁以下の2進数のうち、1の個数がk個のものの個数, 合計を求める"""
    if k == 0:
        return 1, 0
    if n < k:  # n=0含む
        return 0, 0
    # 個数は、n桁のうちk桁を選ぶ組み合わせ
    count = comb(n, k)
    # 各桁に対して、他のn-1桁からk-1桁を選ぶ組み合わせの回数
    sum = (pow(2, n, mod) - 1) * comb(n-1, k-1)
    if mod is None:
        return count, sum
    return count % mod, sum % mod
