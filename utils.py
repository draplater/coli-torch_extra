from collections import Counter, namedtuple
import re
from itertools import chain, islice

import dynet as dn


class ConllEntry(object):
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats, str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])

    def __hash__(self):
        return self.id


class ArcHybridState(object):
    def __init__(self, sentence, options):
        self.sentence = sentence
        self.next_word = 0
        self.stack = []
        self.lstm_map = None
        self.total_feature_map = {}

        self.options = options
        self.nnvecs = (1 if self.options.headFlag else 0) + \
                 (2 if self.options.rlFlag or self.options.rlMostFlag else 0)

    def set_lstm_map(self, lstm_map):
        self.lstm_map = lstm_map

        for root in self.sentence:
           self.total_feature_map[root] =  [self.lstm_map[root] for _ in xrange(self.nnvecs)]

    def copy_for_foresee(self):
        other = self.__class__(self.sentence, self.options)
        other.stack = list(self.stack)
        other.next_word = self.next_word
        other.total_feature_map = self.total_feature_map
        return other

    @property
    def is_stack_empty(self):
        return len(self.stack) == 0

    @property
    def is_buffer_empty(self):
        return self.next_word == len(self.sentence)

    @property
    def is_buffer_root(self):
        return self.next_word == len(self.sentence) - 1

    def is_finished(self):
        return len(self.stack) == 0 and self.is_buffer_root

    def get_stack_top_k(self, k, empty_pad):
        return [self.total_feature_map[self.stack[-i-1]]
                if len(self.stack) > i else [empty_pad] for i in xrange(k)]

    @property
    def next_word_root(self):
        return self.sentence[self.next_word]

    def get_buffer_top(self, empty_pad):
        if self.is_buffer_empty:
            return [empty_pad]
        return [self.total_feature_map[self.next_word_root]]

    def get_input_tensor(self, k, empty_pad):
        topStack = self.get_stack_top_k(k, empty_pad)
        topBuffer = self.get_buffer_top(empty_pad)
        return dn.concatenate(list(chain(*(topStack + topBuffer))))

    @property
    def alpha(self):
        return islice(self.stack, 0, len(self.stack) - 2) if len(self.stack) > 2 else iter(())

    @property
    def s1(self):
        return islice(self.stack, len(self.stack) - 2, len(self.stack) - 1) if len(self.stack) > 1 else iter(())

    @property
    def s0(self):
        return islice(self.stack, len(self.stack) - 1, None) if len(self.stack) > 0 else iter(())

    @property
    def b(self):
        return islice(self.sentence, self.next_word, self.next_word + 1) if not self.is_buffer_empty else iter(())

    @property
    def beta(self):
        return islice(self.sentence, self.next_word + 1, None) \
            if len(self.sentence) - self.next_word > 1 else iter(())

class ArcHybridTransition(object):
    pass


class ArcHybridActions(list):
    class ARC_LEFT(ArcHybridTransition):
        require_relation = True
        @staticmethod
        def can_do_action(state):
            """
            :type state: ArcHybridState
            """
            return not state.is_stack_empty and not state.is_buffer_empty

        @staticmethod
        def do_action(state, relation):
            """
            :type state: ArcHybridState
            """
            child = state.stack.pop()
            parent = state.next_word_root

            child.pred_parent_id = parent.id
            child.pred_relation = relation

            bestOp = 0
            hoffset = 1 if state.options.headFlag else 0
            if state.options.rlMostFlag:
                state.total_feature_map[parent][bestOp + hoffset] = state.total_feature_map[child][bestOp + hoffset]
            if state.options.rlFlag:
                state.total_feature_map[parent][bestOp + hoffset] = state.lstm_map[child]

        @staticmethod
        def is_correct(state, relation, with_relation=True):
            stack_back = state.stack[-1]
            ret = all(h.id != stack_back.parent_id for h in chain(state.s1, state.beta)) and \
                  all(d.parent_id != stack_back.id for d in chain(state.b, state.beta))
            if with_relation:
                ret = (ret and relation == stack_back.relation)
            return ret

        def __str__(self):
            return "ARC_LEFT"

    class ARC_RIGHT(ArcHybridTransition):
        require_relation = True
        @staticmethod
        def can_do_action(state):
            """
            :type state: ArcHybridState
            """
            return len(state.stack) >= 2 and state.stack[-1].id != 0

        @staticmethod
        def do_action(state, relation):
            """
            :type state: ArcHybridState
            """
            child = state.stack.pop()
            parent = state.stack[-1]

            child.pred_parent_id = parent.id
            child.pred_relation = relation

            bestOp = 1
            hoffset = 1 if state.options.headFlag else 0
            if state.options.rlMostFlag:
                state.total_feature_map[parent][bestOp + hoffset] = state.total_feature_map[child][bestOp + hoffset]
            if state.options.rlFlag:
                state.total_feature_map[parent][bestOp + hoffset] = state.lstm_map[child]

        @staticmethod
        def is_correct(state, relation, with_relation=True):
            stack_back = state.stack[-1]
            ret = all(h.id != stack_back.parent_id and \
                      h.parent_id != stack_back.id for h in chain(state.b, state.beta))
            if with_relation:
                ret = (ret and relation == stack_back.relation)
            return ret

        def __str__(self):
            return "ARC_RIGHT"

    class SHIFT(ArcHybridTransition):
        require_relation = False
        @staticmethod
        def can_do_action(state):
            """
            :type state: ArcHybridState
            """
            return not state.is_buffer_empty and state.next_word_root.id != 0

        @staticmethod
        def do_action(state, relation):
            """
            :type state: ArcHybridState
            """
            state.stack.append(state.next_word_root)
            state.next_word += 1

        @staticmethod
        def is_correct(state, relation, with_relation=True):
            return all(h.id != state.next_word_root.parent_id for h in chain(state.s1, state.alpha)) and \
            all(d.parent_id != state.next_word_root.id for d in state.stack)

        def __str__(self):
            return "SHIFT"

    class INVALID(ArcHybridTransition):
        @staticmethod
        def can_do_action(state):
            return False

        @staticmethod
        def do_action(state, relation):
            """
            :type state: ArcHybridState
            """
            raise RuntimeError("Invalid action.")

        @staticmethod
        def is_correct(state, relation, with_relation=True):
            return False

    class Combo(ArcHybridTransition):
        def __init__(self, seq):
            self.action_seq = list(seq) # type: [ArcHybridTransition]

            require_relation_count = sum(1 for i in self.action_seq if i.require_relation)
            if require_relation_count == 0:
                self.require_relation = False
            elif require_relation_count == 1:
                self.require_relation = True
            else:
                raise NotImplementedError

        def can_do_action(self, state):
            current_state = state # type: ArcHybridState
            for action in self.action_seq:
                if not action.can_do_action(current_state):
                    return False
                current_state = current_state.copy_for_foresee()
                action.do_action(current_state, "XXX")
            return True

        def do_action(self, state, relation):
            for action in self.action_seq:
                action.do_action(state, relation)

        def is_correct(self, state, relation, with_relation=True):
            current_state = state # type: ArcHybridState
            for action in self.action_seq:
                if not action.is_correct(current_state, relation,
                                         with_relation=with_relation):
                    return False
                current_state = current_state.copy_for_foresee()
                action.do_action(current_state, relation)
            return True

        def __str__(self):
            return "Combo({})".format(",".join(str(i) for i in self.action_seq))

        def __repr__(self):
            return self.__str__()

    ActionWithRelation = namedtuple("ActionWithRelation", ["action", "relation",
                                                           "action_index", "relation_index"])

    def __init__(self, relations, action_file):
        if action_file is None:
            actions = [self.SHIFT, self.ARC_LEFT, self.ARC_RIGHT]
        else:
            actions = self.generate_actions(action_file)

        super(ArcHybridActions, self).__init__(actions)

        print "Actions: {}".format(self)
        self.relations = relations
        self.decoded_with_relation = []

        for action_idx, action in enumerate(self):
            if not action.require_relation:
                self.decoded_with_relation.append(
                    self.ActionWithRelation(action, None, action_idx, -1))

        for relation_idx, relation in enumerate(self.relations):
            for action_idx, action in enumerate(self):
                if action.require_relation:
                    self.decoded_with_relation.append(
                        self.ActionWithRelation(action, relation, action_idx, relation_idx))

    @classmethod
    def generate_actions(cls, action_file):
        str_to_action = {"arc_left": cls.ARC_LEFT, "arc_right": cls.ARC_RIGHT,
                         "shift": cls.SHIFT}
        with open(action_file) as f:
            for line in f:
                if not line:
                    continue
                actions = [str_to_action[i] for i in line.strip().split()]
                if len(actions) == 1:
                    yield actions[0]
                else:
                    yield cls.Combo(actions)

class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            # root.scores = None
            root.parent = None
            root.pred_parent_id = 0 # None
            root.pred_relation = 'rroot' # None
            # root.vecs = None
            # root.lstms = None

    def __len__(self):
        return len(self.roots)


    def Attach(self, parent_index, child_index):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        del self.roots[child_index]


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id]-=1
                forest.Attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
                unassigned[forest.roots[i].id]-=1
                forest.Attach(i, i+1)
                break

    return len(forest.roots) == 1


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP, True):
            wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
            relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), relCount.keys())


def read_conll(fh, proj):
    dropped = 0
    read = 0
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1:
                if not proj or isProj([t for t in tokens if isinstance(t, ConllEntry)]):
                    yield tokens
                else:
                    #print 'Non-projective sentence dropped'
                    dropped += 1
                read += 1
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        yield tokens

    print dropped, 'dropped non-projective sentences.'
    print read, 'sentences read.'


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

cposTable = {"PRP$": "PRON", "VBG": "VERB", "VBD": "VERB", "VBN": "VERB", ",": ".", "''": ".", "VBP": "VERB", "WDT": "DET", "JJ": "ADJ", "WP": "PRON", "VBZ": "VERB", 
             "DT": "DET", "#": ".", "RP": "PRT", "$": ".", "NN": "NOUN", ")": ".", "(": ".", "FW": "X", "POS": "PRT", ".": ".", "TO": "PRT", "PRP": "PRON", "RB": "ADV", 
             ":": ".", "NNS": "NOUN", "NNP": "NOUN", "``": ".", "WRB": "ADV", "CC": "CONJ", "LS": "X", "PDT": "DET", "RBS": "ADV", "RBR": "ADV", "CD": "NUM", "EX": "DET", 
             "IN": "ADP", "WP$": "PRON", "MD": "VERB", "NNPS": "NOUN", "JJS": "ADJ", "JJR": "ADJ", "SYM": "X", "VB": "VERB", "UH": "X", "ROOT-POS": "ROOT-CPOS", 
             "-LRB-": ".", "-RRB-": "."}
