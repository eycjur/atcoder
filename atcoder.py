# 実行 ctrl + F5
# デバッグ F5

import bisect
import heapq
import itertools
import math
import sys
from collections import Counter, defaultdict, deque
from functools import cache

from more_itertools import distinct_permutations


# 1行&複数文字を分割して取得
a, b = [int(x) for x in input().split()]
# 1行&複数文字をリストとして取得
l = [int(x) for x in input().split()]
# 1行&単一文字列を一文字ごとに分割して取得
a = list(input())
# 複数行&一文字をリストとして取得
N = int(input())
a = [int(input()) for _ in range(N)]
# 複数行&複数文字をリストのリストとして取得
N = int(input())
a = [[int(x) for x in input().split()] for _ in range(N)]
# 複数行&複数文字を列ごとにリストで取得
N = int(input())
ab = [[int(x) for x in input().split()] for _ in range(N)]
a, b = [list(i) for i in zip(*ab)]
# 複数行&複数文字を最初の列と他の列で分けてリストで取得
N = int(input())
k_s = [list(map(int, input().split())) for _ in range(N)]
k = [x[0] for x in k_s]
s = [x[1:] for x in k_s]


# 二分探索
class SortedList:
    def __init__(self, list_: list | None = None) -> None:
        """ソート済みリスト

        Args:
            list_ (list | None, optional): 初期値. Defaults to [].
        """
        if list_ is not None:
            # Note: とりうる値が限られている場合はビンソート（O(N)）を検討
            self._list = sorted(list_)
        else:
            self._list = []

    def __contains__(self, value: int) -> bool:
        """値が存在するかどうかを判定

        Example:
            >>> sl = SortedList([1, 3, 5])
            >>> 3 in sl
            True
            >>> 4 in sl
            False
        """
        idx = bisect.bisect_left(self._list, value)
        return idx < len(self._list) and self._list[idx] == value

    def insert(self, value: int) -> None:
        """順序を保ちつつ値を挿入

        Example:
            >>> sl = SortedList([1, 3, 5])
            >>> sl.insert(4)
            >>> sl
            SortedList([1, 3, 4, 5])
        """
        bisect.insort_left(self._list, value)

    def bisect_left(self, value: int) -> int:
        """挿入位置を返す（同じ値がある場合は左側）

        「挿入する値よりも小さい値がいくつあるか」を返す

        Example:
            >>> sl = SortedList([1, 3, 5])
            >>> sl.bisect_left(3)
            1
            >>> sl.bisect_left(4)
            2
            >>> sl.bisect_left(6)
            3
        """
        return bisect.bisect_left(self._list, value)

    def bisect_right(self, value: int) -> int:
        """挿入位置を返す（同じ値がある場合は右側）

        「挿入する値よりも大きい値がいくつあるか」を返す

        Example:
            >>> sl = SortedList([1, 3, 5])
            >>> sl.bisect_right(3)
            2
            >>> sl.bisect_right(4)
            2
        """
        return bisect.bisect_right(self._list, value)


def binary_search(func: Callable[[int], bool], bottom: int, top: int) -> int:
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
        >>> binary_search(func, 0, 100)
        10

    Note:
        最小値を探索する場合は、本関数内のtopとbottomを入れ替える
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


# 優先度付きキュー
from typing import TypeVar
T = TypeVar("T")

class HeapQ:
    def __init__(self, list_: list[T] | None = None) -> None:
        """優先度付きキューを初期化する

        Args:
            list_ (list[T], optional): 初期値. Defaults to [].
        """
        self._heap = list_ or []
        heapq.heapify(self._heap)

    def push(self, x: T) -> None:
        """要素を追加する"""
        heapq.heappush(self._heap, x)

    def pop(self) -> T:
        """最小値を取り出す"""
        return heapq.heappop(self._heap)

    def __bool__(self) -> bool:
        """要素があるかどうか"""
        return bool(self._heap)


class UnionFind:
    def __init__(self, n: int) -> None:
        """UnionFindを初期化する

        Args:
            n (int): 要素数
        """
        self._n = n
        # 親要素の番号を格納する。-1の場合は自分が根
        self._parents = [-1] * n
        # グループのサイズ（根のみが有効な値を持つ）
        self._sizes = [1] * n
        # グループにおける木の深さ（根のみが有効な値を持つ）
        self._rank = [0] * n

    def find_root(self, i: int) -> int:
        """i番目の要素の根を返す

        Args:
            i (int): 要素

        Returns:
            int: 根
        """
        if self._parents[i] == -1:
            return i
        else:
            self._parents[i] = self.find_root(self._parents[i]) # 経路圧縮
            return self._parents[i]

    def unite(self, i: int, j: int) -> None:
        """要素iと要素jの属するグループを併合する

        Args:
            i (int): 要素
            j (int): 要素
        """
        root_i = self.find_root(i)
        root_j = self.find_root(j)

        if root_i == root_j:
            return

        # ランクの低い木を高い木に併合
        if self._rank[root_i] < self._rank[root_j]:
            self._parents[root_i] = root_j
            # sizeの更新は根のみでよい
            self._sizes[root_j] += self._sizes[root_i]
        else:
            self._parents[root_j] = root_i
            self._sizes[root_i] += self._sizes[root_j]
            # ランクが同じなら片方を1つ上げる
            if self._rank[root_i] == self._rank[root_j]:
                # ランクの更新は根のみでよい
                self._rank[root_i] += 1

    def size(self, i: int) -> int:
        """i番目の要素が属するグループのサイズを返す

        Args:
            i (int): 要素

        Returns:
            int: グループのサイズ
        """
        return self._sizes[self.find_root(i)]

    def rank(self, i: int) -> int:
        """i番目の要素が属するグループの木の深さを返す

        Args:
            i (int): 要素

        Returns:
            int: 木の深さ
        """
        return self._rank[self.find_root(i)]

    def same(self, i: int, j: int) -> bool:
        """要素iと要素jが同じグループに属するかを返す

        Args:
            i (int): 要素
            j (int): 要素

        Returns:
            bool: 同じグループに属するか
        """
        return self.find_root(i)==self.find_root(j)

    def members(self, i: int) -> list[int]:
        """i番目の要素が属するグループに属する要素のリストを返す

        Args:
            i (int): 要素

        Returns:
            list[int]: グループに属する要素のリスト
        """
        root = self.find_root(i)
        return [j for j in range(self._n) if self.find_root(j) == root]

    @property
    def roots(self) -> list[int]:
        """根のリストを返す

        Returns:
            list[int]: 根のリスト
        """
        return [i for i, x in enumerate(self._parents) if x < 0]

    @property
    def group_count(self) -> int:
        """グループの数を返す

        Returns:
            int: グループの数
        """
        return len(self.roots)

    @property
    def all_group_members(self) -> dict[int, list[int]]:
        """全てのグループのメンバーを返す

        Returns:
            dict[int, list[int]]: グループのメンバー
        """
        return {r: self.members(r) for r in self.roots}

    def __str__(self) -> str:
        return ', '.join(f'{r}: {m}' for r, m in self.all_group_members.items())
