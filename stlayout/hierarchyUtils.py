import networkx as nx


def clean_idle_nodes(indexedNodes):
    # elimate nodes with too few children(less than 3)
    nodeIndexes = sorted(indexedNodes.keys())
    for nid in nodeIndexes:
        node = indexedNodes[nid]
        if not node['ancIdx']:
            continue
        # not-leaves node but have children less than 3
        if node['virtualNode'] and len(node['childIdx']) < 3:
            ancNode = indexedNodes[node['ancIdx']]
            ancNodeChildren = set(ancNode['childIdx'])
            ancNodeChildren.remove(nid)

            for cid in node['childIdx']:
                indexedNodes[cid]['ancIdx'] = ancNode['idx']
                ancNodeChildren.add(cid)

            ancNode['childIdx'] = list(ancNodeChildren)
            del indexedNodes[nid]
    return indexedNodes


def set_nodes_height(indexedNodes, root, graph):
    # refine height for every node
    # top-down
    queue = [root]
    indexedNodes[graph['rootIdx']]['height'] = 0
    minHeight = 0
    while len(queue) > 0:
        nid = queue.pop(0)
        nHeight = indexedNodes[nid]['height']
        for c in indexedNodes[nid]['childIdx']:
            indexedNodes[c]['height'] = nHeight - 1
            queue.append(c)
        if len(indexedNodes[nid]['childIdx']) > 0:
            if nHeight - 1 < minHeight:
                minHeight = nHeight - 1
    maxHeight = abs(minHeight)

    for i, node in indexedNodes.items():
        node['height'] += maxHeight
    return indexedNodes, maxHeight


def get_ancestor_chain(nid, indexedNodes, explore_list):
    if indexedNodes[nid]['ancIdx']:
        explore_list.append(indexedNodes[nid]['ancIdx'])
        return get_ancestor_chain(indexedNodes[nid]['ancIdx'], indexedNodes, explore_list)
    else:
        return explore_list


# Least Common Ancestor
def get_LCA(source, target, indexedNodes):
    sourceList = indexedNodes[source]['ancestors']
    targetList = set(indexedNodes[target]['ancestors'])
    for a in sourceList:
        if a in targetList:
            return a
    return None