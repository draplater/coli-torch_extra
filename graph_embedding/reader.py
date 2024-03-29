import json
import pickle
import torch
from collections import defaultdict

import dataclasses
from dataclasses import dataclass, field

from coli.basic_tools.common_utils import AttrDict
from coli.data_utils.dataset import PAD
from coli.hrgguru.hrg import CFGRule
from coli.hrgguru.hyper_graph import strip_category, HyperEdge
from coli.data_utils.vocab_utils import Dictionary


@dataclass
class GraphEmbeddingStatisticsBase(object):
    words: Dictionary = field(
        default_factory=Dictionary,
        metadata={"embedding_dim": 100}
    )

    conn_labels: Dictionary = field(
        default_factory=Dictionary,
        metadata={"embedding_dim": 100}
    )

    use_out = False

    @classmethod
    def get_properties(cls):
        return [i.name for i in
                dataclasses.fields(cls)
                if i.name != "words" and i.name != "conn_labels"]

    @classmethod
    def get_embedding_dim_of(cls, prop_name):
        # noinspection PyUnresolvedReferences
        return cls.__dataclass_fields__[prop_name].metadata["embedding_dim"]

    def read_sync_rule_and_update(self, sync_rule: CFGRule,
                                  allow_new=True):
        raise NotImplementedError


@dataclass
class GraphEmbeddingStatistics(GraphEmbeddingStatisticsBase):
    postags: Dictionary = field(
        default_factory=Dictionary,
        metadata={"embedding_dim": 100}
    )
    senses: Dictionary = field(
        default_factory=Dictionary,
        metadata={"embedding_dim": 100}
    )

    external_index: Dictionary = field(
        default_factory=Dictionary,
        metadata={"embedding_dim": 10}
    )

    is_lf = False

    def read_sync_rule_and_update(self, sync_rule: CFGRule,
                                  allow_new=True):
        statistics = self
        rule = sync_rule.hrg
        assert rule is not None

        left_edge = right_edge = None
        if sync_rule.rhs is not None:
            if isinstance(sync_rule.rhs[0][0], str) and isinstance(sync_rule.rhs[0][1], HyperEdge):
                left_edge = sync_rule.rhs[0][1]
            if len(sync_rule.rhs) >= 2 and isinstance(sync_rule.rhs[1][1], HyperEdge):
                right_edge = sync_rule.rhs[1][1]

        node_to_pred_edge = {edge.nodes[0]: edge
                             for edge in rule.rhs.edges
                             if len(edge.nodes) == 1}

        if self.is_lf:
            arg0_to_pred_edge = {}
            lbl_to_pred_edge = {}
            for edge in rule.rhs.edges:
                if edge.label == "ARG0":
                    arg0_to_pred_edge[edge.nodes[1]] = node_to_pred_edge[edge.nodes[0]]
                if edge.label == "LBL":
                    lbl_to_pred_edge[edge.nodes[1]] = node_to_pred_edge[edge.nodes[0]]

        nodes_sorted = sorted(rule.rhs.nodes, key=lambda x: int(x.name))

        ret = AttrDict(
            entities=torch.zeros(len(rule.rhs.nodes), dtype=torch.int64),
            external_indices=torch.zeros(len(rule.lhs.nodes), dtype=torch.int64),
            properties=[torch.zeros(len(rule.rhs.nodes), dtype=torch.int64)
                        for _ in range(len(self.get_properties()))],
            conn_indices=[[] for _ in range(len(rule.rhs.nodes))],
            conn_labels=[[] for _ in range(len(rule.rhs.nodes))],
        )

        if self.use_out:
            ret.out_conn_indices = [[] for _ in range(len(rule.rhs.nodes))]
            ret.out_conn_labels = [[] for _ in range(len(rule.rhs.nodes))]

        # left
        if left_edge is not None:
            # noinspection PyCallingNonCallable
            ret.left_nodes = torch.tensor([nodes_sorted.index(i) for i in left_edge.nodes])
        else:
            ret.left_nodes = None

        # right
        if right_edge is not None:
            # noinspection PyCallingNonCallable
            ret.right_nodes = torch.tensor([nodes_sorted.index(i) for i in right_edge.nodes])
        else:
            ret.right_nodes = None

        for node_id, node in enumerate(nodes_sorted):
            assert int(node.name) == node_id
            try:
                external_index_int = rule.lhs.nodes.index(node)
                external_index = str(external_index_int)
                ret.external_indices[external_index_int] = node_id
            except ValueError:
                external_index = "-1"
            pred_edge = node_to_pred_edge.get(node)
            lemma = pos = sense = "None"
            if left_edge is not None and pred_edge == left_edge:
                lemma = "__LEFT__"
            elif right_edge is not None and pred_edge == right_edge:
                lemma = "__RIGHT__"
            elif pred_edge is not None:
                assert pred_edge.is_terminal
                lemma, pos, sense = strip_category(pred_edge.label, return_tuple=True)
            node_int = statistics.words.update_and_get_id(lemma, allow_new=allow_new)
            ret.entities[node_id] = node_int
            ret.properties[0][node_id] = statistics.postags.update_and_get_id(
                pos, allow_new=allow_new)
            ret.properties[1][node_id] = statistics.senses.update_and_get_id(
                sense, allow_new=allow_new)
            ret.properties[2][node_id] = statistics.external_index.update_and_get_id(
                external_index, allow_new=allow_new)

            if self.is_lf:
                node_type = PAD
                pred_edge_as_arg0 = arg0_to_pred_edge.get(node)
                pred_edge_as_lbl = lbl_to_pred_edge.get(node)
                pred_edge = None
                if pred_edge_as_arg0:
                    node_type = "ARG0"
                    pred_edge = pred_edge_as_arg0
                elif pred_edge_as_lbl:
                    node_type = "LBL"
                    pred_edge = pred_edge_as_lbl
                if pred_edge is not None:
                    lemma, pos, sense = strip_category(pred_edge.label, return_tuple=True)
                    node_int = statistics.words.update_and_get_id(lemma, allow_new=allow_new)
                    ret.entities[node_id] = node_int
                    ret.properties[0][node_id] = statistics.postags.update_and_get_id(
                        pos, allow_new=allow_new)
                    ret.properties[1][node_id] = statistics.senses.update_and_get_id(
                        sense, allow_new=allow_new)
                # noinspection PyUnresolvedReferences
                ret.properties[3][node_id] = statistics.types.update_and_get_id(
                    node_type, allow_new=False)

            ret.conn_indices[node_id].append(node_id)
            ret.conn_labels[node_id].append(statistics.conn_labels.update_and_get_id("Self", allow_new=allow_new))

            if self.use_out:
                ret.out_conn_indices[node_id].append(node_id)
                ret.out_conn_labels[node_id].append(
                    statistics.conn_labels.update_and_get_id("Self", allow_new=allow_new))

            for edge in rule.rhs.edges:
                if len(edge.nodes) >= 2 and node == edge.nodes[0]:
                    label = edge.label
                    if edge == left_edge:
                        label = "__LEFT__"
                    elif edge == right_edge:
                        label = "__RIGHT__"
                    else:
                        assert edge.is_terminal
                    for other_node in edge.nodes[1:]:
                        other_node_id = int(other_node.name)
                        ret["conn_indices" if not self.use_out else "out_conn_indices"][node_id].append(other_node_id)
                        ret["conn_labels" if not self.use_out else "out_conn_labels"][node_id].append(
                            statistics.conn_labels.update_and_get_id(
                                label, allow_new=allow_new))
                        ret.conn_indices[other_node_id].append(node_id)
                        ret.conn_labels[other_node_id].append(
                            statistics.conn_labels.update_and_get_id(
                                label, allow_new=allow_new))

        keys = ["conn_indices", "conn_labels"]
        if self.use_out:
            keys.extend(["out_conn_indices", "out_conn_labels"])
        for key in keys:
            for i in range(len(ret[key])):
                # noinspection PyCallingNonCallable
                ret[key][i] = torch.tensor(ret[key][i])
            # ret[key] = pad_and_stack_1d(ret[key])

        return ret

    def read_subgraph_and_update(self, sync_rule: CFGRule,
                                 allow_new=True):
        statistics = self
        rule = sync_rule.hrg
        assert rule is not None

        left_edge = right_edge = None
        if sync_rule.rhs is not None:
            if isinstance(sync_rule.rhs[0][1], HyperEdge):
                left_edge = sync_rule.rhs[0][1]
            if len(sync_rule.rhs) >= 2 and isinstance(sync_rule.rhs[1][1], HyperEdge):
                right_edge = sync_rule.rhs[1][1]

        node_to_pred_edge = {edge.nodes[0]: edge
                             for edge in rule.rhs.edges
                             if len(edge.nodes) == 1}
        ret = {
            'entities': [],
            'properties': [[] for _ in range(len(self.get_properties()))],
            'conn_indices': [[] for _ in range(len(rule.rhs.nodes))],
            'conn_labels': [[] for _ in range(len(rule.rhs.nodes))],
        }

        if self.use_out:
            ret["out_conn_indices"] = [[] for _ in range(len(rule.rhs.nodes))]
            ret["out_conn_labels"] = [[] for _ in range(len(rule.rhs.nodes))]

        for node_id, node in enumerate(sorted(rule.rhs.nodes, key=lambda x: int(x.name))):
            assert int(node.name) == node_id
            try:
                external_index = str(rule.lhs.nodes.index(node))
            except ValueError:
                external_index = "-1"
            pred_edge = node_to_pred_edge.get(node)
            lemma = pos = sense = PAD
            if pred_edge == left_edge:
                lemma = "__LEFT__"
            elif pred_edge == right_edge:
                lemma = "__RIGHT__"
            elif pred_edge is not None:
                assert pred_edge.is_terminal
                lemma, pos, sense = strip_category(pred_edge.label, return_tuple=True)
            node_int = statistics.words.update_and_get_id(lemma, allow_new=allow_new)
            ret["entities"].append(node_int)
            ret["properties"][0].append(statistics.postags.update_and_get_id(pos, allow_new=allow_new))
            ret["properties"][1].append(statistics.senses.update_and_get_id(sense, allow_new=allow_new))
            ret["properties"][2].append(
                statistics.external_index.update_and_get_id(external_index, allow_new=allow_new))

            ret["conn_indices"][node_id].append(node_id)
            ret["conn_labels"][node_id].append(statistics.conn_labels.update_and_get_id("Self", allow_new=allow_new))

            if self.use_out:
                ret["out_conn_indices"][node_id].append(node_id)
                ret["out_conn_labels"][node_id].append(
                    statistics.conn_labels.update_and_get_id("Self", allow_new=allow_new))

            for edge in rule.rhs.edges:
                if len(edge.nodes) >= 2 and node == edge.nodes[0]:
                    label = edge.label
                    if edge == left_edge:
                        label = "__LEFT__"
                    elif edge == right_edge:
                        label = "__RIGHT__"
                    else:
                        assert edge.is_terminal
                    for other_node in edge.nodes[1:]:
                        other_node_id = int(other_node.name)
                        ret["conn_indices" if not self.use_out else "out_conn_indices"][node_id].append(other_node_id)
                        ret["conn_labels" if not self.use_out else "out_conn_labels"][node_id].append(
                            statistics.conn_labels.update_and_get_id(
                                label, allow_new=allow_new))
                        ret["conn_indices"][other_node_id].append(node_id)
                        ret["conn_labels"][other_node_id].append(
                            statistics.conn_labels.update_and_get_id(
                                label, allow_new=allow_new))
        return ret


class GraphEmbeddingStatisticsBiDir(GraphEmbeddingStatistics):
    use_out = True


@dataclass
class GraphEmbeddingStatisticsLF(GraphEmbeddingStatistics):
    use_out = True
    is_lf = True

    types: Dictionary = field(
        default_factory=Dictionary,
        metadata={"embedding_dim": 10}
    )


@dataclass
class GraphEmbeddingStatisticsDual(GraphEmbeddingStatisticsBase):
    postag_and_senses: Dictionary = field(
        default_factory=Dictionary,
        metadata={"embedding_dim": 100}
    )
    conn_count: Dictionary = field(
        default_factory=Dictionary,
        metadata={"embedding_dim": 10}
    )

    def read_sync_rule_and_update(self, sync_rule: CFGRule, allow_new=True):
        statistics = self
        rule = sync_rule.hrg
        assert rule is not None

        left_edge = right_edge = None
        if sync_rule.rhs is not None:
            if isinstance(sync_rule.rhs[0][1], HyperEdge):
                left_edge = sync_rule.rhs[0][1]
            if len(sync_rule.rhs) >= 2 and isinstance(sync_rule.rhs[1][1], HyperEdge):
                right_edge = sync_rule.rhs[1][1]

        ret = {
            'entities': [],
            'properties': [[] for _ in range(len(self.get_properties()))],
            'conn_indices': [[] for _ in range(len(rule.rhs.edges))],
            'conn_labels': [[] for _ in range(len(rule.rhs.edges))],
        }

        edge_by_node = defaultdict(list)
        for edge_id, edge in enumerate(rule.rhs.edges):
            for idx, node in enumerate(edge.nodes):
                edge_by_node[node].append((idx, edge_id))

        for edge_id, edge in enumerate(rule.rhs.edges):
            pos = sense = "None"
            if edge.is_terminal and len(edge.nodes) == 1:
                lemma, pos, sense = strip_category(edge.label, return_tuple=True)
            elif edge.is_terminal and len(edge.nodes) == 2:
                lemma = edge.label
            else:
                assert not edge.is_terminal
                lemma = edge.label.split("+++")[0]
                if edge == left_edge:
                    pos = "__LEFT__"
                else:
                    assert edge == right_edge
                    pos = "__RIGHT__"
            edge_int = statistics.words.update_and_get_id(lemma, allow_new=allow_new)
            ret["entities"].append(edge_int)
            ret["properties"][0].append(
                statistics.postag_and_senses.update_and_get_id(pos + "_" + sense, allow_new=allow_new))
            ret["properties"][1].append(statistics.conn_count.update_and_get_id(len(edge.nodes), allow_new=allow_new))

            for node_idx, node in enumerate(edge.nodes):
                for other_node_idx, other_edge_id in edge_by_node[node]:
                    if other_edge_id == edge_id:
                        conn_label = "Self"
                    else:
                        idx1, idx2 = sorted([node_idx, other_node_idx])
                        ext = "E" if node in rule.lhs.nodes else "I"
                        conn_label = "{}-{}-{}".format(idx1, idx2, ext)
                    ret["conn_indices"][edge_id].append(other_edge_id)
                    ret["conn_labels"][edge_id].append(
                        statistics.conn_labels.update_and_get_id(conn_label, allow_new=allow_new))

        return ret


graph_embedding_types = {"normal": GraphEmbeddingStatistics,
                         "bidir": GraphEmbeddingStatisticsBiDir,
                         "lf": GraphEmbeddingStatisticsLF,
                         "dual": GraphEmbeddingStatisticsDual}

if __name__ == '__main__':
    import os

    name = "deepbank1.1-lfrg2-small-qeq-181015"
    home = os.path.expanduser("~")
    grammar_file = home + f"/Development/HRGGuru/deepbank-preprocessed/cfg_hrg_mapping-{name}.pickle"
    simple_graphs = []
    statistics = GraphEmbeddingStatistics()
    with open(grammar_file, "rb") as f:
        grammar = pickle.load(f)
    for rules_and_counts in grammar.values():
        for rule in rules_and_counts.keys():
            if rule.hrg is not None:
                simple_graph = statistics.read_sync_rule_and_update(rule)
                simple_graphs.append((rule, simple_graph))
                print(json.dumps(simple_graph))
                pass
    pass
