from typing import List

def expand_list_to(indexes: List[int], m: int) -> List[int]:
    r = indexes.copy()

    for i in range(m):
        if i not in indexes:
            r.append(i)

    return r