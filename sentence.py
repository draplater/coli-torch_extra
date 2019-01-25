import torch
from dataclasses import field, dataclass
from torch.nn import Module, ModuleDict, Embedding, LayerNorm

from coli.basic_tools.dataclass_argparse import argfield, OptionsBase
from coli.torch_extra.layers import CharacterEmbedding
from coli.torch_span.layers import FeatureDropout, LayerNormalization


class SentenceEmbeddings(Module):
    @dataclass
    class Options(OptionsBase):
        dim_word: "word embedding dim" = 100
        dim_postag: "postag embedding dim. 0 for not using postag" = 100
        dim_char: "character embedding dim. 0 for not using character" = 100
        word_dropout: "word embedding dropout" = 0.4
        postag_dropout: "postag embedding dropout" = 0.2
        character_embedding: CharacterEmbedding.Options = field(
            default_factory=CharacterEmbedding.Options)
        input_layer_norm: "Use layer norm on input embeddings" = True
        mode: str = argfield("concat", choices=["add", "concat"])

    def __init__(self,
                 hparams: "SentenceEmbeddings.Options",
                 statistics,
                 plugins=None
                 ):

        super().__init__()
        self.hparams = hparams
        self.mode = hparams.mode
        self.plugins = ModuleDict(plugins) if plugins is not None else {}

        # embedding
        input_dims = []
        if hparams.dim_word != 0:
            self.word_embeddings = Embedding(
                len(statistics.words), hparams.dim_word, padding_idx=0)
            self.word_dropout = FeatureDropout(hparams.word_dropout)
            input_dims.append(hparams.dim_word)

        if hparams.dim_postag != 0:
            self.pos_embeddings = Embedding(
                len(statistics.postags), hparams.dim_postag, padding_idx=0)
            self.pos_dropout = FeatureDropout(hparams.postag_dropout)
            input_dims.append(hparams.dim_postag)

        if "pretrained_contextual" in self.plugins:
            input_dims.append(self.plugins["pretrained_contextual"].output_dim)
            self.character_lookup = self.char_embeded = None
        elif hparams.dim_char > 0:
            self.bilm = None
            self.character_lookup = Embedding(len(statistics.characters), hparams.dim_char)
            self.char_embeded = CharacterEmbedding.get(hparams.character_embedding, input_size=hparams.dim_char)
            input_dims.append(hparams.dim_char)
        else:
            self.bilm = None
            self.character_lookup = self.char_embeded = None

        if hparams.mode == "concat":
            self.output_dim = sum(input_dims)
        else:
            assert hparams.mode == "add"
            uniq_input_dims = list(set(input_dims))
            if len(uniq_input_dims) != 1:
                raise ValueError(f"Different input dims: {uniq_input_dims}")
            self.output_dim = uniq_input_dims[0]

        self.input_layer_norm = LayerNorm(self.output_dim, eps=1e-6) \
            if hparams.input_layer_norm else None

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.word_embeddings.weight.data)
        if self.hparams.dim_postag != 0:
            torch.nn.init.xavier_normal_(self.pos_embeddings.weight.data)
        if self.character_lookup is not None:
            torch.nn.init.xavier_normal_(self.character_lookup.weight.data)

    def forward(self, inputs):
        # input embedding
        # word

        if self.hparams.dim_word != 0:
            words = inputs.words
            word_embeded = self.word_embeddings(words)

            if "external_embedding" in self.plugins:
                external_embedding = self.plugins["external_embedding"]
                pretrained_word_embeded = external_embedding(inputs)
                word_embeded += pretrained_word_embeded
        else:
            word_embeded = None

        # pos
        if self.hparams.dim_postag != 0:
            pos_embeded = self.pos_embeddings(inputs.postags)
        else:
            pos_embeded = None

        # character
        # batch_size, bucket_size, word_length, embedding_dims
        if "pretrained_contextual" in self.plugins:
            pretrained_contextual = self.plugins["pretrained_contextual"]
            word_embeded_by_char = pretrained_contextual(inputs)
        elif self.hparams.dim_char:
            # use character embedding instead
            # batch_size, bucket_size, word_length, embedding_dims
            char_embeded_4d = self.character_lookup(inputs.chars)
            word_embeded_by_char = self.char_embeded(inputs.word_lengths,
                                                     char_embeded_4d)
        else:
            word_embeded_by_char = None

        all_features = list(filter(lambda x: x is not None,
                            [word_embeded, pos_embeded, word_embeded_by_char]))

        if self.mode == "concat":
            total_input_embeded = torch.cat(all_features, -1)
        else:
            total_input_embeded = sum(all_features)

        if self.input_layer_norm is not None:
            total_input_embeded = self.input_layer_norm(total_input_embeded)

        return total_input_embeded
