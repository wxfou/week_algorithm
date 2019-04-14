# -*- coding:utf-8 -*-
"""
Description:双向字典树
迭代次数默认最大999。

@author: wuxiaofei
@date: 2019/4/15
"""


class TrieNode(object):
    def __init__(self, value=None, count=0, parent=None):
        self.value = value
        self.count = count
        self.parent = parent
        self.children = {}


class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, sequence):
        cur_node = self.root
        for item in sequence:
            if item not in cur_node.children:
                child = TrieNode(value=item, count=1, parent=cur_node)
                cur_node.children[item] = child
                cur_node = child
            else:
                cur_node = cur_node.children[item]
                cur_node.count += 1

    def search(self, sequence):
        cur_node = self.root
        mark = True
        for item in sequence:
            if item not in cur_node.children:
                mark = False
                break
            else:
                cur_node = cur_node.children[item]
        if cur_node.children:
            mark = False
        return mark

    def delete(self, sequence):
        mark = False
        if self.search(sequence):
            mark = True
            cur_node = self.root
            for item in sequence:
                cur_node.children[item].count -= 1
                if cur_node.children[item].count == 0:
                    cur_node.children.pop(item)
                    break
                else:
                    cur_node = cur_node.children[item]
        return mark

    def search_part(self, sequence, prefix, suffix, start_node=None):
        if start_node:
            cur_node = start_node
            prefix_node = start_node.parent
        else:
            cur_node = self.root
            prefix_node = self.root
        mark = True
        for i in range(len(sequence)):
            if i == 0:
                if sequence[i] != cur_node.value:
                    for child_node in cur_node.children.values():
                        self.search_part(sequence, prefix, suffix, child_node)
                    mark = False
                    break
            else:
                if sequence[i] not in cur_node.children:
                    for child_node in cur_node.children.values():
                        self.search_part(sequence, prefix, suffix, child_node)
                    mark = False
                    break
                else:
                    cur_node = cur_node.children[sequence[i]]
        if mark:
            if prefix_node.value:
                if prefix_node.value in prefix:
                    prefix[prefix_node.value] += cur_node.count
                else:
                    prefix[prefix_node.value] = cur_node.count
            for suffix_node in cur_node.children.values():
                if suffix_node.value in suffix:
                    suffix[suffix_node.value] += suffix_node.count
                else:
                    suffix[suffix_node.value] = suffix_node.count
            for child_node in cur_node.children.values():
                self.search_part(sequence, prefix, suffix, child_node)


if __name__ == "__main__":
    trie = Trie()
    texts = [["小米", "华为", "oppo", "vivo", "apple", "哈哈"], ["oppo", "华为", "vivo"], ["1", "2", "3", "4"], ["some", "one", "go"],
             ["you", "are", "爱"]]
    for text in texts:
        trie.insert(text)
    flag = trie.search(["oppo", "华为", "vivo"])
    print(flag)
    flag = trie.search(["忘记", "了"])
    print(flag)
    flag = trie.search(["忘记", "爱"])
    print(flag)
    flag = trie.delete(["oppo", "少年"])
    print(flag)
    prefixx = {}
    suffixx = {}
    trie.search_part(["华为"], prefixx, suffixx)
    print(prefixx)
    print(suffixx)
