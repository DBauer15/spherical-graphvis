import networkx as nx
import json
import math
import matplotlib.pyplot as plt

from stlayout.hierarchyUtils import get_ancestor_chain, get_LCA
from copy import deepcopy


def get_weight(item, indexedNodes):
    return indexedNodes[item]['size']


class TreeMap:
    def __init__(self, data_path, width, height):
        self.indexedNodes = {}
        self.graph = None
        self.root = None
        self.hierarchy = {}
        self.links_in_node = None
        self.quotient_graphs = {}
        self.origin_graph = None

        with open(data_path) as fp:
            self.graph = json.load(fp)
            self.root = self.graph['rootIdx']
            for node in self.graph['nodes']:
                self.indexedNodes[node['idx']] = node
                self.indexedNodes[node['idx']]['size'] = 1

            self.indexedNodes[self.root]['rect'] = {
                'width': width,
                'height': height,
                'x': -width / 2,
                'y': -height / 2
            }
            self.indexedNodes[self.root]['pos2D'] = (0, 0)
            self.origin_graph = self.construct_nx_graph()

    def top_down_search(self, nid, do_something):
        do_something(nid)
        if self.indexedNodes[nid]['virtualNode']:
            for item in self.indexedNodes[nid]['childIdx']:
                self.top_down_search(item, do_something)
        else:
            pass

    def construct_nx_graph(self, draw=False):
        g = nx.Graph()
        for node in self.graph['nodes']:
            if not node['virtualNode']:
                g.add_node(node['idx'])
        for edge in self.graph['links']:
            g.add_edge(edge['sourceIdx'], edge['targetIdx'])
        if draw:
            nx.draw(g, node_size=50, with_labels=False, node_color="blue", alpha=0.5, edge_color="gray")
        return g

    def build_hierarchy(self, maxHeight):
        hierarchy = {}
        for i in range(maxHeight + 1):
            hierarchy[i] = []
        for i, node in self.indexedNodes.items():
            hierarchy[node['height']].append(i)
        self.hierarchy = hierarchy

    def construct_quotient_graph(self):
        self.quotient_graphs = {}
        for l, nodes in self.hierarchy.items():
            if l == 0:
                continue
            for nidx in nodes:
                if not self.indexedNodes[nidx]['virtualNode']:
                    continue
                node = self.indexedNodes[nidx]
                g = nx.Graph()
                for c in node['childIdx']:
                    g.add_node(c, size=self.indexedNodes[c]['size'])
                links = self.links_in_node[nidx]
                for link, weight in links.items():
                    g.add_edge(link[0], link[1], weight=math.log(weight))
                size = 0
                for n in g.nodes():
                    size += g.nodes[n]['size']
                self.indexedNodes[nidx]['size'] = size
                self.quotient_graphs[nidx] = g

    def layout_quotient_graphs(self, layout):
        for idx, g in self.quotient_graphs.items():
            try:
                positions = layout(self.quotient_graphs[idx], pos=None, iterations=2000)
            except ZeroDivisionError:
                print('given layout method failed, using spring_layout in networkx')
                positions = nx.spring_layout(self.quotient_graphs[idx])
            g.layout = positions

    def find_links_in_super_nodes(self):
        for i, node in self.indexedNodes.items():
            if not node['virtualNode']:
                node['ancestors'] = get_ancestor_chain(i, self.indexedNodes, [])
        # construct links in quotient graphs
        linksInNodes = {}
        for link in self.graph['links']:
            lca = get_LCA(link['sourceIdx'], link['targetIdx'], self.indexedNodes)
            if lca not in linksInNodes:
                linksInNodes[lca] = {}
            source = link['sourceIdx']
            target = link['targetIdx']
            for i in self.indexedNodes[source]['ancestors']:
                if i == lca:
                    break
                else:
                    source = i
            for i in self.indexedNodes[target]['ancestors']:
                if i == lca:
                    break
                else:
                    target = i
            if (source, target) not in linksInNodes[lca]:
                linksInNodes[lca][(source, target)] = 0
            linksInNodes[lca][(source, target)] += 1
        self.links_in_node = linksInNodes

    def accumulate_weight(self, l):
        weight = 0
        for item in l:
            weight += get_weight(item, self.indexedNodes)
        return weight

    def split(self, top_down_list, left_right_list, rect, weight):
        if len(left_right_list) == 0:
            return

        if len(left_right_list) == 1:
            self.indexedNodes[left_right_list[0]]['rect'] = rect
            return
        half_weight = weight / 2

        if rect['width'] > rect['height']:
            # using left-right split
            l_1, w1, l_2, w2 = self.weight_split(left_right_list, half_weight)
            r1 = {
                'width': rect['width'] * w1 / weight,
                'height': rect['height'],
                'x': rect['x'],
                'y': rect['y']
            }
            r2 = {
                'width': rect['width'] - r1['width'],
                'height': rect['height'],
                'x': rect['x'] + r1['width'],
                'y': rect['y']
            }
            topdown_l1 = [i for i in top_down_list if i in l_1]
            topdown_l2 = [i for i in top_down_list if i in l_2]
            self.split(topdown_l1, l_1, r1, w1)
            self.split(topdown_l2, l_2, r2, w2)
        else:
            # using left-right split
            l_1, w1, l_2, w2 = self.weight_split(top_down_list, half_weight)
            r1 = {
                'width': rect['width'],
                'height': rect['height'] * w1 / weight,
                'x': rect['x'],
                'y': rect['y']
            }
            r2 = {
                'width': rect['width'],
                'height': rect['height'] - r1['height'],
                'x': rect['x'],
                'y': rect['y'] + r1['height']
            }
            leftright_l1 = [i for i in left_right_list if i in l_1]
            leftright_l2 = [i for i in left_right_list if i in l_2]
            self.split(l_1, leftright_l1, r1, w1)
            self.split(l_2, leftright_l2, r2, w2)

    def weight_split(self, l, half_weight):
        w1 = 0
        tmp = 0
        l_1, l_2 = [], []
        for item in l:
            tmp = w1 + get_weight(item, self.indexedNodes)
            if abs(half_weight - tmp) > abs(half_weight - w1):
                break
            l_1.append(item)
            w1 = tmp
        l_2 = []
        w2 = 0
        for item in l:
            if item in l_1:
                continue
            else:
                l_2.append(item)
                w2 += get_weight(item, self.indexedNodes)
        return l_1, w1, l_2, w2

    def treemap_layout(self, nid):
        rect = self.indexedNodes[nid]['rect']
        cx = rect['x'] + rect['width'] / 2
        cy = rect['y'] + rect['height'] / 2
        self.indexedNodes[nid]['pos2D'] = (cx, cy)
        # TODO clean
        # TODO this condition statement have logical issues
        if not self.indexedNodes[self.indexedNodes[nid]['childIdx'][0]]['virtualNode']:
            size = min(rect['width'], rect['height']) / 2
            r_list = [math.sqrt(pos[0] * pos[0] + pos[1] * pos[1]) for i, pos in
                      self.quotient_graphs[nid].layout.items()]
            pos_list = [pos for i, pos in self.quotient_graphs[nid].layout.items()]
            mx, my = 0, 0
            for p in pos_list:
                mx += p[0]
                my += p[1]
            mx, my = mx / len(pos_list), my / len(pos_list)
            max_radius = max(r_list)
            scaling_ratio = float(size) / max_radius
            for c in self.indexedNodes[nid]['childIdx']:
                dx = self.quotient_graphs[nid].layout[c][0] - mx
                dy = self.quotient_graphs[nid].layout[c][1] - my
                self.indexedNodes[c]['pos2D'] = (cx + dx * scaling_ratio, cy + dy * scaling_ratio)
            return
        rect = self.indexedNodes[nid]['rect']
        # sort the children
        top_down_list = deepcopy(self.indexedNodes[nid]['childIdx'])
        top_down_list = sorted(top_down_list, key=lambda x: self.quotient_graphs[nid].layout[x][1])
        left_right_list = deepcopy(self.indexedNodes[nid]['childIdx'])
        left_right_list = sorted(left_right_list, key=lambda x: self.quotient_graphs[nid].layout[x][0])
        weight = self.accumulate_weight(top_down_list)
        self.split(top_down_list, left_right_list, rect, weight)
        for c in self.indexedNodes[nid]['childIdx']:
            self.treemap_layout(c)

    def draw(self):
        position_2d = {}
        for i, node in self.indexedNodes.items():
            if not node['virtualNode']:
                position_2d[node['idx']] = node['pos2D']
        nx.draw_networkx_nodes(self.origin_graph, position_2d, node_size=50, with_labels=False, node_color="blue", alpha=0.5)
        nx.draw_networkx_edges(self.origin_graph, position_2d, edge_color="gray", alpha=0.5)
        plt.axis('off')
        plt.show()
        return position_2d

    def save_layout(self, path):
        json_obj = {
            "nodes": [],
            "links": self.graph['links'],
            "rootIdx": self.root
        }
        for node in self.graph['nodes']:
            if node['idx'] in self.indexedNodes:
                json_obj['nodes'].append(node)
        with open(path, 'w') as fp:
            json.dump(json_obj, fp)