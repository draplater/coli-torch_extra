import torch
from allennlp.nn.util import get_dropout_mask

from torch import nn, Tensor
import torch.nn.init as I
import torch.nn.functional as F
from torch.nn import LayerNorm

from coli.torch_extra.seq_utils import sort_sequences, unsort_sequences, pad_timestamps_and_batches
from coli.torch_extra.utils import BatchIndices


class TFCompatibleLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size,
                 weight_initializer=I.xavier_normal_,
                 forget_bias=1.0,
                 activation=torch.tanh):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_initializer = weight_initializer
        self.activation = activation
        self.forget_bias = forget_bias

        self.linearity = nn.Linear(input_size + self.hidden_size, 4 * hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_initializer(self.linearity.weight)
        I.zeros_(self.linearity.bias)

    def forward(self, inputs, state):
        (h_prev, c_prev) = state
        lstm_matrix = self.linearity(torch.cat([inputs, h_prev], 1))
        i, j, f, o = torch.split(
            lstm_matrix,
            lstm_matrix.shape[1] // 4,
            dim=1)
        c = torch.sigmoid(f + self.forget_bias) * c_prev + \
            torch.sigmoid(i) * self.activation(j)
        h = torch.sigmoid(o) * self.activation(c)
        return h, (h, c)

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(
            self.input_size, self.hidden_size
        )


class DynamicRNN(nn.Module):
    def __init__(self, cell,
                 input_keep_prob=1.0,
                 recurrent_keep_prob=1.0,
                 go_forward=True):
        super().__init__()
        self.cell = cell
        self.input_keep_prob = input_keep_prob
        self.recurrent_keep_prob = recurrent_keep_prob
        self.go_forward = go_forward

        self.hidden_size = self.cell.hidden_size

    def forward(self, sequence_tensor, batch_lengths):
        batch_size = sequence_tensor.shape[0]
        total_timesteps = batch_lengths[0]
        output_accumulator = sequence_tensor.new_zeros(batch_size, total_timesteps, self.hidden_size)
        full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.hidden_size)
        full_batch_previous_state = sequence_tensor.data.new_zeros(batch_size, self.hidden_size)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_keep_prob < 1.0:
            recurrent_dropout_mask = get_dropout_mask(1 - self.recurrent_keep_prob,
                                                      full_batch_previous_memory)
        else:
            recurrent_dropout_mask = None

        if self.input_keep_prob < 1.0:
            sequence_tensor = F.dropout(sequence_tensor, 1 - self.input_keep_prob, self.training)

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            # What we are doing here is finding the index into the batch dimension
            # which we need to use for this timestep, because the sequences have
            # variable length, so once the index is greater than the length of this
            # particular batch sequence, we no longer need to do the computation for
            # this sequence. The key thing to recognise here is that the batch inputs
            # must be _ordered_ by length from longest (first in batch) to shortest
            # (last) so initially, we are going forwards with every sequence and as we
            # pass the index at which the shortest elements of the batch finish,
            # we stop picking them up for the computation.
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
                while current_length_index < (len(batch_lengths) - 1) and \
                        batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            # Actually get the slices of the batch which we need for the computation at this timestep.
            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            timestep_input = sequence_tensor[0: current_length_index + 1, index]

            _, (timestep_output, memory) = self.cell(timestep_input, (previous_state, previous_memory))

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if recurrent_dropout_mask is not None and self.training:
                timestep_output = timestep_output * recurrent_dropout_mask[0: current_length_index + 1]

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, hidden_size). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state


class DynamicRNN1D(DynamicRNN):
    def forward(self, sequence_tensor_1d, batch_indices: BatchIndices, seqs_to_process_list):
        batch_size = batch_indices.batch_size
        total_timesteps = batch_indices.max_len
        output_accumulator = sequence_tensor_1d.new_zeros(sequence_tensor_1d.size(0), self.hidden_size)
        full_batch_previous_memory = sequence_tensor_1d.new_zeros(batch_size, self.hidden_size)
        full_batch_previous_state = sequence_tensor_1d.data.new_zeros(batch_size, self.hidden_size)

        if self.recurrent_keep_prob < 1.0:
            recurrent_dropout_mask = get_dropout_mask(1 - self.recurrent_keep_prob,
                                                      full_batch_previous_memory)
        else:
            recurrent_dropout_mask = None

        if self.input_keep_prob < 1.0:
            sequence_tensor_1d = F.dropout(sequence_tensor_1d, 1 - self.input_keep_prob, self.training)

        for timestep in range(total_timesteps):
            seqs_to_process = seqs_to_process_list[timestep]
            if self.go_forward:
                idxs = batch_indices.startpoints[seqs_to_process] + timestep
            else:
                idxs = batch_indices.endpoints[seqs_to_process] - timestep
            # Actually get the slices of the batch which we need for the computation at this timestep.
            previous_memory = full_batch_previous_memory[seqs_to_process].clone()
            previous_state = full_batch_previous_state[seqs_to_process].clone()
            timestep_input = sequence_tensor_1d[idxs]

            _, (timestep_output, memory) = self.cell(timestep_input, (previous_state, previous_memory))

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if recurrent_dropout_mask is not None and self.training:
                timestep_output = timestep_output * recurrent_dropout_mask[seqs_to_process]

            # We've been doing computation with less than the full batch, so here we create a new
            # variable for the the whole batch at this timestep and insert the result for the
            # relevant elements of the batch into it.
            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[seqs_to_process] = memory
            full_batch_previous_state[seqs_to_process] = timestep_output
            output_accumulator[idxs] = timestep_output

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, hidden_size). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state


class LSTM(torch.nn.Module):
    """
    A standard stacked Bidirectional LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular bidirectional LSTM is the application of
    variational dropout to the hidden states of the LSTM.
    Note that this will be slower, as it doesn't use CUDNN.
    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM (for each direction).
    num_layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    """

    cell_class = TFCompatibleLSTMCell
    rnn_class = DynamicRNN

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 input_keep_prob: float = 1.0,
                 recurrent_keep_prob: float = 1.0,
                 layer_norm=False
                 ) -> None:
        super(LSTM, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            forward_layer = self.rnn_class(
                self.cell_class(lstm_input_size, hidden_size),
                input_keep_prob,
                recurrent_keep_prob,
                go_forward=True)
            backward_layer = self.rnn_class(
                self.cell_class(lstm_input_size, hidden_size),
                input_keep_prob,
                recurrent_keep_prob,
                go_forward=False)
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)

            if layer_norm:
                layer_norm_ = LayerNorm(hidden_size * 2)
                self.add_module('layer_norm_{}'.format(layer_index), layer_norm_)
            else:
                layer_norm_ = None
            layers.append([forward_layer, backward_layer, layer_norm_])
            lstm_input_size = hidden_size * 2
        self.lstm_layers = layers

    def forward(self,  # pylint: disable=arguments-differ
                seqs: Tensor,
                lengths: Tensor,
                is_sorted=False
                ):
        original_shape = seqs.shape
        if not is_sorted:
            seqs, lengths, unsort_idx = sort_sequences(seqs, lengths, False)
        output_seqs = seqs
        for i, (forward_layer, backward_layer, layer_norm) in enumerate(self.lstm_layers):
            forward_output, final_forward_state = forward_layer(output_seqs, lengths)
            backward_output, final_backward_state = backward_layer(output_seqs, lengths)
            output_seqs = torch.cat([forward_output, backward_output], -1)
            if layer_norm is not None:
                output_seqs = layer_norm(output_seqs)
        if not is_sorted:
            # noinspection PyUnboundLocalVariable
            output_seqs = unsort_sequences(output_seqs, unsort_idx, original_shape, unpack=False)
        else:
            # noinspection PyUnboundLocalVariable
            output_seqs = pad_timestamps_and_batches(output_seqs, original_shape)
        # noinspection PyUnboundLocalVariable
        return output_seqs


class LSTM1D(LSTM):
    rnn_class = DynamicRNN1D
    # noinspection PyMethodOverriding
    def forward(self,  # pylint: disable=arguments-differ
                seqs: Tensor,
                batch_indices,
                return_all=True
                ):
        total_timesteps = batch_indices.max_len
        seqs_to_process_list = []
        for timestep in range(total_timesteps):
            seqs_to_process = torch.gt(batch_indices.seq_lens, timestep)
            seqs_to_process_list.append(seqs_to_process)

        outputs_each_layer = []
        output_seqs = seqs
        for i, (forward_layer, backward_layer, layer_norm) in enumerate(self.lstm_layers):
            forward_output, final_forward_state = forward_layer(output_seqs, batch_indices, seqs_to_process_list)
            backward_output, final_backward_state = backward_layer(output_seqs, batch_indices, seqs_to_process_list)
            output_seqs = torch.cat([forward_output, backward_output], -1)
            if layer_norm is not None:
                output_seqs = layer_norm(output_seqs)
            outputs_each_layer.append(output_seqs)
        if not return_all:
            # noinspection PyUnboundLocalVariable
            return output_seqs
        else:
            return torch.stack(outputs_each_layer, dim=1)
