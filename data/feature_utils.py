from typing import List
import torch


def one_hot_encoding(value: int, choices: List) -> List:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def subgraph_index(G, n):
    allpaths = []
    for node in G:
        paths = findPaths2(G, node , n)
        allpaths.extend(paths)
    allpaths = torch.tensor(allpaths, dtype=torch.long).T
    return allpaths

def findPaths2(G,u,n,excludeSet = None):
    if excludeSet == None:
        excludeSet = set([u])
    else:
        excludeSet.add(u)
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet for path in findPaths2(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)
    return paths