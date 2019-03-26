import torch

from coli.torch_extra.graph_embedding.padding import sequence_mask
from coli.torch_extra.utils import pad_and_stack_1d, pad_and_stack_2d_2


class Batch:
    def init(self, options, instances, device="cpu"):
        self.size = len(instances)
        # shape: [batch_size, num_entities]
        self.entity_labels = pad_and_stack_1d([i.entities for i in instances], device=device)
        # shape: [batch_size, num_entities]
        # noinspection PyCallingNonCallable
        entities_mask = sequence_mask(
            torch.tensor([len(i.entities) for i in instances], device=device))
        self.entities_mask = entities_mask.float().unsqueeze(-1)

        if options.use_property_embeddings:
            # list of shape: [batch_size, num_entities]
            self.entity_properties_list = [
                pad_and_stack_1d([i.properties[prop_idx] for i in instances],
                                 device=device)
                for prop_idx in range(len(instances[0].properties))
            ]

        # if options.use_char_embedding:
        #     # shape: [batch_size, num_entities, chars_per_word]
        #     self.entity_chars = pad_3d_values([_['entity_chars'] for _ in instances])
        #     # shape: [batch_size, num_entities]
        #     self.entity_chars_nums = pad_2d_values([[len(chars) for chars in _['entity_chars']]
        #                                             for _ in instances])

        # shape: [batch_size, num_entities, incoming_degree]
        # noinspection PyCallingNonCallable
        conn_mask = sequence_mask(pad_and_stack_1d([
            torch.tensor([len(indices) for indices in i.conn_indices])
            for i in instances
        ], device=device))

        self.conn_mask = conn_mask.float().unsqueeze(-1)

        # shape: [batch_size, num_entities, incoming_degree]
        self.conn_indices = pad_and_stack_2d_2([i.conn_indices for i in instances],
                                             device=device)
        # shape: [batch_size, num_entities, incoming_degree]
        self.conn_labels = pad_and_stack_2d_2([i.conn_labels for i in instances],
                                            device=device)

        if "out_conn_indices" in instances[0]:
            # shape: [batch_size, num_entities, out_degree]
            # noinspection PyCallingNonCallable
            out_conn_mask = sequence_mask(pad_and_stack_1d([
                torch.tensor([len(indices) for indices in i.out_conn_indices])
                for i in instances
            ], device=device))

            self.out_conn_mask = out_conn_mask.float().unsqueeze(-1)

            # shape: [batch_size, num_entities, out_degree]
            self.out_conn_indices = pad_and_stack_2d_2([i.out_conn_indices for i in instances],
                                                     device=device)
            # shape: [batch_size, num_entities, out_degree]
            self.out_conn_labels = pad_and_stack_2d_2([i.out_conn_labels for i in instances],
                                                    device=device)
