from abc import ABC
from os import path

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertConfig, Trainer, get_scheduler, BertModel
from torch.optim import AdamW
import argparse
from tqdm import tqdm
import pickle
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from deepspeed.runtime.engine import DeepSpeedEngine


def print_rank_0(message, local_rank, debug=True):
    if local_rank == 0 and debug:
        print(message)


def _add_core_arguments(parser):
    r"""Helper (internal) function to update an argument parser with an argument group of the core DeepSpeed arguments.
        The core set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.
        This is a helper function to the public add_config_arguments()
    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    group = parser.add_argument_group('DeepSpeed', 'DeepSpeed configurations')

    group.add_argument(
        '--deepspeed',
        default=False,
        action='store_true',
        help='Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepspeed_config',
                       default=None,
                       type=str,
                       help='DeepSpeed json configuration file.')

    return parser


def add_config_arguments(parser):
    r"""Update the argument parser to enabling parsing of DeepSpeed command line arguments.
        The set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.
    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    parser = _add_core_arguments(parser)

    return parser


class BertPreTrainingHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size, vocab_size, hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.output_bias

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sequence_output = self.decoder(sequence_output)
        return sequence_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act=nn.GELU()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        return sequence_output


class BertLargeCls(nn.Module):
    def __init__(self, config):
        super().__init__()

        mlm_criterion = nn.CrossEntropyLoss(reduction="none")
        self.max_predictions_per_seq = config.max_predictions_per_seq

        def get_masked_lm_loss(
                logit_blob,
                masked_lm_positions,
                masked_lm_labels,
                label_weights,
                max_predictions_per_seq,
        ):
            # gather valid position indices
            logit_blob = torch.gather(
                logit_blob,
                index=masked_lm_positions.unsqueeze(2).to(
                    dtype=torch.int64).repeat(1, 1, 30522),
                dim=1,
            )
            logit_blob = torch.reshape(logit_blob, [-1, 30522])
            label_id_blob = torch.reshape(masked_lm_labels, [-1])

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            pre_example_loss = mlm_criterion(logit_blob, label_id_blob.long())
            pre_example_loss = torch.reshape(
                pre_example_loss, [-1, max_predictions_per_seq])
            sum_label_weight = torch.sum(label_weights, dim=-1)
            sum_label_weight = sum_label_weight / label_weights.shape[0]
            numerator = torch.sum(pre_example_loss * label_weights)
            denominator = torch.sum(label_weights) + 1e-5
            loss = numerator / denominator
            return loss

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config.hidden_size, config.vocab_size)
        self.ns_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.masked_lm_criterion = get_masked_lm_loss

    def forward(self, labels, id, pos, weight, inputs, return_outputs=False):
        outputs = self.bert(**inputs)
        prediction_scores, seq_relationship_scores = self.cls(
            outputs[0], outputs[1])  # last_hidden_state, pooler_output
        next_sentence_loss = self.ns_criterion(
            seq_relationship_scores.view(-1, 2), labels.long().view(-1)
        )
        masked_lm_loss = self.masked_lm_criterion(
            prediction_scores, pos, id, weight, max_predictions_per_seq=self.max_predictions_per_seq
        )

        total_loss = next_sentence_loss + masked_lm_loss
        return (total_loss, outputs) if return_outputs else total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Training batch size"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=24, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=16,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=512, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("--max_predictions_per_seq", type=int, default=80)
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument('--local_rank', type=int, default=-1,
                        help="local rank passed from distributed launcher")

    parser = add_config_arguments(parser)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    configuration = BertConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               num_attention_heads=args.num_attention_heads, intermediate_size=4 * args.hidden_size,
                               max_predictions_per_seq=args.max_predictions_per_seq, local_rank=args.local_rank)

    model = BertLargeCls(configuration)

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.adam_weight_decay)
    lr_scheduler = get_scheduler(
        "polynomial",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=300
    )
    # print_rank_0("model structure\n {}".format(model), args.local_rank)

    # Data Prepare
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')

    # DeepSpeed
    # model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
    #                                                      model=model,
    #                                                      model_parameters=filter(lambda p: p.requires_grad,
    #                                                                              model.parameters()))

    model_engine = DeepSpeedEngine(args=args,
                                   model=model,
                                   optimizer=optimizer,
                                   model_parameters=filter(lambda p: p.requires_grad,
                                                           model.parameters()),
                                   training_data=None,
                                   lr_scheduler=lr_scheduler,
                                   mpu=None,
                                   dist_init_required=None,
                                   collate_fn=None,
                                   config=None,
                                   config_params=None)

    print_rank_0("After DeepSpeed Initalizing, MemoryAllocated: {}".format(
        torch.cuda.memory_allocated()), args.local_rank)  # 2940059136
    print_rank_0("After DeepSpeed Initalizing, MaxMemoryAllocated: {}".format(
        torch.cuda.max_memory_allocated()), args.local_rank)  # 5144944128 应该是有一半的梯度或者参数被划分了才初始化优化器的。
    print("train_batch_size is:", args.train_batch_size)
    for i in range(args.epochs):
        for _ in range(1024):
            # encoded_input.data, label, id, pos, weight = batch
            encoded_input.data = {}

            encoded_input.data['input_ids'] = torch.tensor(np.random.randint(1, 30522, (args.train_batch_size, 512)),
                                                           dtype=torch.int32).to(
                model_engine.device)  # 这里的1是batchsize，可以调。
            encoded_input.data['token_type_ids'] = torch.tensor(np.random.randint(0, 1, (args.train_batch_size, 512)),
                                                                dtype=torch.int32).to(
                model_engine.device)
            encoded_input.data['attention_mask'] = torch.tensor(np.random.randint(0, 1, (args.train_batch_size, 512)),
                                                                dtype=torch.int32).to(
                model_engine.device)

            label = torch.tensor(np.random.randint(
                0, 1, (args.train_batch_size, 1)), dtype=torch.int32).to(model_engine.device)
            id = torch.tensor(np.random.randint(
                0, 1, (args.train_batch_size, 80)), dtype=torch.int32).to(model_engine.device)
            pos = torch.tensor(np.random.randint(
                0, 1, (args.train_batch_size, 80)), dtype=torch.int32).to(model_engine.device)
            weight = torch.tensor(np.random.randint(
                0, 1, (args.train_batch_size, 80)), dtype=torch.int32).to(model_engine.device)

            loss = model_engine(label, id, pos, weight, encoded_input)
            # loss = model(label, id, pos, weight, encoded_input)
            forward_size = torch.cuda.memory_allocated()
            forward_max_size = torch.cuda.max_memory_allocated()
            model_engine.backward(loss)
            backward_size = torch.cuda.memory_allocated()
            backward_max_size = torch.cuda.max_memory_allocated()
            # loss.backward()

            # optimizer.step()
            # lr_scheduler.step()
            # prof.step()
            # model_engine.step()
            # optimizer.zero_grad()
