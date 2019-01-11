from typing import List
from weakref import WeakValueDictionary

global_elmo_cache = WeakValueDictionary()

def get_elmo(options_file: str,
             weight_file: str,
             num_output_representations: int,
             requires_grad: bool = False,
             do_layer_norm: bool = False,
             dropout: float = 0.5,
             vocab_to_cache: List[str] = None,
             keep_sentence_boundaries: bool = False,
             scalar_mix=None
             ):
    from allennlp.modules import Elmo
    key = (options_file, weight_file)
    old_elmo = global_elmo_cache.get(key)
    if old_elmo:
        # noinspection PyProtectedMember
        module = old_elmo._elmo_lstm
        options_file = None
        weight_file = None
    else:
        module = None

    ret = Elmo(options_file=options_file,
               weight_file=weight_file,
               num_output_representations=num_output_representations,
               requires_grad=requires_grad,
               do_layer_norm=do_layer_norm,
               dropout=dropout,
               vocab_to_cache=vocab_to_cache,
               keep_sentence_boundaries=keep_sentence_boundaries,
               module=module)

    if not old_elmo:
        global_elmo_cache[key] = ret

    return ret

