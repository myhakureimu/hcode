"""Synthetic datasets to test in-context learning ability."""
from typing import Tuple
import os
import torch
import dataclasses
from torch.utils.data import TensorDataset, Dataset, DataLoader
from typing import Dict
import numpy as np
from tqdm import tqdm
from collections import Counter

#from src.dataloaders.base import SequenceDataset
from base import SequenceDataset
from pythomata import SimpleDFA
import random

class DFA:
    """Represents a DFA"""

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        transitions: Tuple[dict],
        rng: np.random.Generator,
    ):
        assert len(transitions) == num_nodes
        transitions = {i: v for i, v in enumerate(transitions)}
        dfa = SimpleDFA(
            states=set(list(range(num_nodes))),
            alphabet=set(alphabet),
            initial_state=0,
            accepting_states=set(list(range(num_nodes))),
            transition_function=transitions,
        )
        self.dfa = dfa
        self.rng = rng

    def _sorted_transitions(self):
        nodes = sorted(list(self.dfa._transition_function.keys()))
        transitions = []
        for node in nodes:
            node_transitions = self.dfa._transition_function[node]
            # sort node transitions by outgoing state
            transitions.append(
                tuple(sorted(node_transitions.items(), key=lambda item: item[1]))
            )
        return tuple(transitions)

    def _minimize(self):
        # minimize super
        self.dfa = self.dfa.minimize()
        return self

    def _trim(self):
        # trim super
        self.dfa = self.dfa.trim()
        return self

    def __hash__(self):
        # Here I assume the initial state is always the smallest node
        return hash(self._sorted_transitions())

    def __call__(self, word: str):
        current_node = self.dfa._initial_state
        for symbol in word.split():
            if symbol not in self.dfa._transition_function[current_node]:
                return False
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
        return True

    def forward(self, word: str):
        current_node = self.dfa._initial_state
        for symbol in word.split():
            if symbol not in self.dfa._transition_function[current_node]:
                return None
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
        return current_node

    def trace(self, word: str):
        current_node = self.dfa._initial_state
        path = [current_node]
        for symbol in word.split():
            try:
                self.dfa._transition_function[current_node]
            except:
                breakpoint()
            if symbol not in self.dfa._transition_function[current_node]:
                return path
            else:
                current_node = self.dfa._transition_function[current_node][symbol]
                path.append(current_node)
        return path

    def sample(self, length=1):
        """Samples a random word from the DFA"""
        current_node = self.dfa._initial_state
        word = ""
        #print('dfa lentgh', length)
        for _ in range(length):
            outgoing_symbols = list(self.dfa._transition_function[current_node].keys())
            symbol = self.rng.choice(outgoing_symbols)
            word += symbol + " "
            #print(symbol)
            current_node = self.dfa._transition_function[current_node][symbol]
        word = word.rstrip()
        return word


class RandomDFASampler:
    """Samples random DFAs given configs"""

    num_nodes: int
    alphabet: Tuple[str]
    max_outgoing_edge: int
    rng: np.random.Generator = None

    def __init__(
        self,
        num_nodes: int,
        alphabet: Tuple[str],
        max_outgoing_edge: int,
        seed: int = 42,
    ):
        self.num_nodes = num_nodes
        self.alphabet = alphabet
        self.max_outgoing_edge = max_outgoing_edge
        self.rng = np.random.default_rng(seed)

    def sample(self):
        transitions = [{} for _ in range(self.num_nodes)]
        for node in range(self.num_nodes):
            num_transitions = self.rng.integers(1, self.max_outgoing_edge)
            transition_symbols = self.rng.choice(
                self.alphabet, size=num_transitions, replace=False
            )
            # exclude self loops
            possible_nodes = [n for n in range(self.num_nodes) if n != node]
            transition_nodes = self.rng.choice(
                possible_nodes, size=num_transitions, replace=False
            )
            transitions[node] = dict(zip(transition_symbols, transition_nodes))
        dfa_rng = np.random.default_rng(self.rng.integers(0, 2**32))
        return DFA(self.num_nodes, self.alphabet, tuple(transitions), dfa_rng)


if __name__ == "__main__":

    def sample_usage():
        dfa_sampler = RandomDFASampler(4, ("a", "b", "c", "d"), 4, seed=2)
        dfa = dfa_sampler.sample()
        word = dfa.sample(length=10)
        print(word)
        word = dfa.sample(length=10)
        print(word)

    sample_usage()


class Vocab:
    """Custom vocab."""

    def __init__(self, vocab_size: int, special_vocabs: Dict):
        # Special tokens hold seperator and noop/pad token etc
        self.special_vocabs = special_vocabs
        # vocab = []
        # i = 0
        # while len(vocab) < vocab_size:
        #     item = chr(i + 97)
        #     if item not in self.special_vocabs.values():
        #         vocab.append(item)
        #     i += 1
        vocab = [chr(v + 65) for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(list(set(vocab + list(self.special_vocabs.values()))))
        self.v2id = {v: i for i, v in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    @property
    def seperator(self):
        return self.special_vocabs["seperator"]

    @property
    def noop(self):
        return self.special_vocabs["noop"]

    @property
    def special_tokens(self):
        return set(self.special_vocabs.values())

    def get_id(self, token: str):
        return self.v2id[token]

    def get_vocab(self, id: int):
        return self.vocab[id]

    def __len__(self):
        return len(self.vocab)


class Tokenizer:
    """Custom Tokenizer for our own vocab."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def tokenize(
        self, text: str, return_tensor: bool = False, mask_input: bool = False
    ):
        input_ids = [self.vocab.get_id(t) for t in text.split()]

        labels = input_ids[1:]
        input_ids = input_ids[:-1]

        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def decode(self, ids: list):
        return " ".join([self.vocab.get_vocab(id) for id in ids])

'''
class SimpleDataset(Dataset):
    def __init__(self, examples, dfas, tokenizer):
        super().__init__()
        self.inputs = examples[0]
        self.targets = examples[1]
        self.dfas = dfas
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.dfas[idx]
'''
class SimpleDataset(Dataset):
    def __init__(self, examples, dfas, tokenizer):
        super().__init__()
        self.icl_ks = examples['icl_ks']
        self.nm_examples_x = examples['nm'][0]
        self.nm_examples_y = examples['nm'][1]
        self.tv_examples_xs = examples['tv'][0]
        self.tv_examples_ys = examples['tv'][1]
        self.dfas = dfas
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.nm_examples_x)

    def __getitem__(self, idx):
        return self.icl_ks[idx], self.nm_examples_x[idx], self.nm_examples_y[idx], self.tv_examples_xs[idx], self.tv_examples_ys[idx], self.dfas[idx]

class ICLDFADataModule(SequenceDataset):
    _name_ = "icl_dfa"

    def __init__(
        self,
        num_dfas: int,
        num_ood_dfas: int,
        num_ood_samples: int,
        num_random_labels: int,
        insert_task_vector: int,
        insert_mode: str,
        num_examples: int,
        num_valid_examples: int,
        num_icl___examples: int,
        vocab_size: int,
        max_num_nodes: int,
        max_num_in_context_examples: int,
        min_num_in_context_examples: int,
        max_outgoing_edges: int,
        max_len_per_example: int,
        min_len_per_example: int,
        number_duplicates_per_epoch: int = 0,
        input_seq_len: int = 1024,
        seed: int = 0,
        batch_size: int = 32,
        split_train_test: bool = False,
        data_dir: str = None,
        *args,
        **kwargs,
    ):
        self.num_dfas = num_dfas
        self.num_ood_dfas = num_ood_dfas
        self.num_ood_samples = num_ood_samples
        self.num_random_labels = num_random_labels
        self.insert_task_vector = insert_task_vector
        self.insert_mode = insert_mode
        self.num_train_examples = num_examples
        self.num_valid_examples = num_valid_examples
        self.num_icl___examples = num_icl___examples
        self.vocab_size = vocab_size
        self.number_duplicates_per_epoch = number_duplicates_per_epoch

        self.batch_size = batch_size
        self.split_train_test = (
            split_train_test  # let the same copy chars appear in train/test
        )
        self.data_dir = data_dir
        self.max_num_nodes = max_num_nodes
        self.max_num_in_context_examples = max_num_in_context_examples
        self.min_num_in_context_examples = min_num_in_context_examples
        self.max_outgoing_edges = max_outgoing_edges
        self.max_len_per_example = max_len_per_example
        self.min_len_per_example = min_len_per_example
        self.input_seq_len = input_seq_len
        self.seed = seed

        if insert_task_vector:
            self.task_token = ">"
            special_vocabs = {"seperator": ";", "noop": ".", "task": self.task_token}
            self.vocab = Vocab(vocab_size - 3, special_vocabs=special_vocabs)
            #print(self.vocab.vocab)
            #print(len(self.vocab.vocab))
        else:
            special_vocabs = {"seperator": ";", "noop": "."}
            self.vocab = Vocab(vocab_size - 2, special_vocabs=special_vocabs)
        self.special_vocabs = special_vocabs
        self.tokenizer = Tokenizer(self.vocab)

        #self.mask_id = vocab_size - 1 
    def num_special(self):
        return len(self.special_vocabs)
    def insert_once(self, s, char, insert_mode):
        parts = s.split()  # Split the string into parts
        # Choose a random index to insert the character (range is from 0 to len(parts)-2)
        #print(parts)
        if insert_mode == 'random':
            index = random.randint(0, len(parts) - 2)
        elif insert_mode == 'last':
            index = len(parts) - 2
        else:
            raise Exception('wrong insert_mode')
        # Insert the character at the chosen index
        parts.insert(index + 1, char)

        return ' '.join(parts)  # Join the parts back into a string
    
    def generate_samples(self, dfa: DFA, num_samples: int, split: str, length = None):
        if split not in ['train', 'valid', 'icl__', 'ood__', 'ood1_', 'ood2_']:
            raise Exception('wrong generate_mode')
        
        samples = []
        for _ in range(num_samples):
            if split in ['train', 'valid']:
                if self.insert_task_vector:
                    length = self.rng.integers(self.min_len_per_example, self.max_len_per_example)
                else:
                    length = self.rng.integers(1, self.max_len_per_example)
            elif split in ['icl__', 'ood__', 'ood1_', 'ood2_']:
                length = length
            
            word = dfa.sample(length=length)
            if self.insert_task_vector:
                if split in ['train', 'valid']:
                    word = self.insert_once(word, self.task_token, self.insert_mode)
                elif split in ['icl__', 'ood__', 'ood1_', 'ood2_']:
                    word = self.insert_once(word, self.task_token, 'last')
            
            samples.append(word)
        
        return samples
            
    def generate_example(self, dfa: DFA, num_samples: int, split: str, length = None):
        if split not in ['train', 'valid', 'icl__', 'ood__', 'ood1_', 'ood2_']:
            raise Exception('wrong generate_mode')
        
        '''
        nm_example = "" # a long sequence with num_examples samples combined with ' | '
        samples = self.generate_samples(dfa, num_samples, split, length)
        #print('samples')
        #print(samples)
        for sample in samples:
            nm_example += (sample + " " + self.special_vocabs['seperator'] + " ")
        nm_example = nm_example[:-3]
        if len(nm_example) > self.input_seq_len:
            raise Exception('nm_example too long')
        '''
        nm_examples = self.generate_samples(dfa, num_samples, split, length) # a list with num_examples samples
        
        tv_examples = self.generate_samples(dfa, num_samples, split, length) # a list with num_examples samples
        #print('tv_examples')
        #print(tv_examples)
        
        #return self.tokenizer.tokenize(nm_example, return_tensor=True), [self.tokenizer.tokenize(tv_example, return_tensor=True) for tv_example in tv_examples]
        return nm_examples, tv_examples
    
    def setup(self, num_icl_samples=10, icl_sample_length=5, mode='save', dfas_file='GGGG',stage=None):
        #if hasattr(self, "dataset"):
        #    return

        self.rng = np.random.default_rng(self.seed)
        
        total_num = self.num_train_examples + self.num_valid_examples + self.num_icl___examples
        total_ood_num = self.num_icl___examples
        
        DFAs = set([])
        if self.num_dfas == 0:
            raise Exception('not supported')
        else:
            total_num_dfas = self.num_dfas + self.num_ood_dfas
        
        if mode == 'save':
            for _ in tqdm(np.arange(int(total_num_dfas*1.1))):
                num_nodes = self.rng.integers(
                    self.max_outgoing_edges, self.max_num_nodes + 1
                )
                num_alphabet = self.rng.integers(
                    self.max_outgoing_edges, self.vocab_size - self.num_special() + 1
                )
                
                alphabet = self.rng.choice(
                    self.vocab_size - self.num_special(), size=num_alphabet, replace=False
                )
                alphabet = tuple((chr(a + 65) for a in alphabet))
                
                '''
                alphabet = np.random.choice(self.vocab.vocab, size=num_alphabet, replace=False)
                #print(alphabet)
                '''
                sampler = RandomDFASampler(
                    num_nodes,
                    alphabet,
                    self.max_outgoing_edges,
                )
                
                sampler.rng = np.random.default_rng(self.rng.integers(0, 2**32))
                dfa = sampler.sample()
                dfa._minimize()._trim()
                DFAs.add(dfa)
                
                if len(DFAs) >= total_num_dfas:
                    break
                    
            DFAs = list(DFAs)
            self.rng.shuffle(DFAs)
            import pickle
            print('SAVE')
            with open(dfas_file+'.pkl', 'wb') as file:
                pickle.dump(DFAs, file)
        if mode == 'load':
            import pickle
            print('LOAD')
            with open(dfas_file+'.pkl', 'rb') as file:
                DFAs = pickle.load(file)
                
        #print(DFAs)
        DFAs     = random.choices(DFAs[                  :self.num_dfas], k=total_num)
        ood_DFAs = random.choices(DFAs[-self.num_ood_dfas:             ], k=total_ood_num)
        
        '''
        if len(DFAs) < total:
            print(
                "Warning: not enough unique DFAs generated. Using all generated DFAs."
            )
            # scale back
            self.num_train_examples = (len(DFAs) * self.num_train_examples) // total
            self.num_valid_examples = (len(DFAs) * self.num_valid_examples) // total
            self.num_icl___examples = (len(DFAs) * self.num_icl___examples) // total
            self.num_ood___examples = (len(DFAs) - self.num_train_examples - self.num_valid_examples - self.num_icl___examples)
            print(
                f"New num_train_examples: {self.num_train_examples}"
                f"New num_valid_examples: {self.num_valid_examples}"
                f"New num_icl___examples: {self.num_icl___examples}"
                f"New num_ood___examples: {self.num_ood___examples}"
            )
        '''
        
        DFAs = {
            "train": DFAs[                         :  self.num_train_examples],
            "valid": DFAs[ self.num_train_examples : -self.num_icl___examples],
            "icl__": DFAs[-self.num_icl___examples :                         ],
            "ood__": ood_DFAs,
            "ood1_": ood_DFAs,
            "ood2_": ood_DFAs,
        }

        examples = {"train": [], "valid": [], "icl__": [], "ood__": [], "ood1_": [], "ood2_": []}
        icl_examples = []
        connection = " " + self.special_vocabs['seperator'] + " "
        for split, dfas in DFAs.items():
            
            split_examples = []
            print(split)
            for idx, dfa in tqdm(enumerate(dfas), total=len(dfas)):
                if split in ['ood__', 'ood1_']:
                    num_samples = num_icl_samples
                    nm_examples, tv_examples = self.generate_example(dfa, num_samples, split, icl_sample_length)
                    nm_examples = nm_examples[:self.num_ood_samples] + icl_examples[idx]['nm_examples'][self.num_ood_samples:]
                    tv_examples = tv_examples[:self.num_ood_samples] + icl_examples[idx]['tv_examples'][self.num_ood_samples:]
                    if split == 'ood1_': # random input
                        np.random.shuffle(nm_examples)
                        np.random.shuffle(tv_examples)
                elif split == 'ood2_': # random label
                    num_samples = num_icl_samples
                    nm_examples = icl_examples[idx]['nm_examples'].copy()
                    tv_examples = icl_examples[idx]['tv_examples'].copy()
                    for k in range(self.num_random_labels):
                        nm_examples[k] = nm_examples[k][:-1] + random.choice(self.vocab.non_special_vocab)
                        #nm_examples[k] = nm_examples[k][:-1] + random.choice(self.vocab.non_special_vocab)
                        #tv_examples[k] = tv_examples[k][:-1] + random.choice(self.vocab.non_special_vocab) #need to random?
                    np.random.shuffle(nm_examples)
                    np.random.shuffle(tv_examples)
                elif split == "icl__":
                    num_samples = num_icl_samples
                    nm_examples, tv_examples = self.generate_example(dfa, num_samples, split, icl_sample_length)
                    icl_examples.append({'nm_examples': nm_examples, 'tv_examples': tv_examples})
                    
                else:
                    num_samples = self.rng.integers(self.min_num_in_context_examples, self.max_num_in_context_examples)
                    nm_examples, tv_examples = self.generate_example(dfa, num_samples, split)
                    
                nm_example  = connection.join(nm_examples)
                nm_example  = self.tokenizer.tokenize(nm_example, return_tensor=True)
                tv_examples = [self.tokenizer.tokenize(tv_example, return_tensor=True) for tv_example in tv_examples]
            
                #print('nm_example')
                #print(nm_example)
                #print('tv_examples')
                #print(tv_examples)
                #print(example["input_ids"].shape, example["labels"].shape)
                nm_x  = nm_example["input_ids"]
                nm_y  = nm_example["labels"]
                tv_xs = [tv_example["input_ids"] for tv_example in tv_examples]
                tv_ys = [tv_example["labels"   ] for tv_example in tv_examples]
                
                split_examples.append({'icl_k':num_samples,
                                       'nm':(nm_x , nm_y ),
                                       'tv':(tv_xs, tv_ys)
                                       })
            
            
            
            # pad nm examples to same length
            examples_icl_k = [example['icl_k'] for example in split_examples]
            nm_examples_x = torch.nn.utils.rnn.pad_sequence(
                [example['nm'][0] for example in split_examples],
                batch_first=True,
                padding_value=self.vocab.get_id(self.vocab.noop),
            )
            nm_examples_y = torch.nn.utils.rnn.pad_sequence(
                [example['nm'][1] for example in split_examples],
                batch_first=True,
                padding_value=0,
            )
            
            cuts = [0]
            for example in split_examples:
                cuts.append(cuts[-1]+len(example['tv'][0]))
            # pad tv examples to same length
            tv_exampless_x = torch.nn.utils.rnn.pad_sequence(
                [item for example in split_examples for item in example['tv'][0]],
                batch_first=True,
                padding_value=self.vocab.get_id(self.vocab.noop),
            )
    
            tv_exampless_y = torch.nn.utils.rnn.pad_sequence(
                [item for example in split_examples for item in example['tv'][1]],
                batch_first=True,
                padding_value=self.vocab.get_id(self.vocab.noop),
            )
            
            '''
            example_outputs[example_outputs == self.vocab.get_id(";")] = -99
            example_outputs[example_outputs == self.vocab.get_id(">")] = -98
            example_outputs_tv[example_outputs_tv == self.vocab.get_id(";")] = -99
            example_outputs_tv[example_outputs_tv == self.vocab.get_id(">")] = -98
            '''
            tv_examples_xs = []
            tv_examples_ys = []
            for tv_idx in range(len(cuts)-1):
                tv_examples_xs.append(tv_exampless_x[cuts[tv_idx]:cuts[tv_idx+1]])
                tv_examples_ys.append(tv_exampless_y[cuts[tv_idx]:cuts[tv_idx+1]])
            
            #print('************')
            #print(example_inputs.shape)
            #print('************')
            #for example_inputs_tv in example_inputs_tvs:
            #    print(example_inputs_tv.shape)
    
            examples[split] = {'icl_ks': examples_icl_k,
                               'nm': (nm_examples_x , nm_examples_y ),
                               'tv': (tv_examples_xs, tv_examples_ys)}
            
            # origin
            '''
            # pad examples to same length
            example_inputs = torch.nn.utils.rnn.pad_sequence(
                [example[0] for example in split_examples],
                batch_first=True,
                padding_value=self.vocab.get_id(self.vocab.noop),
            )

            example_outputs = torch.nn.utils.rnn.pad_sequence(
                [example[1] for example in split_examples],
                batch_first=True,
                padding_value=self.vocab.get_id(self.vocab.noop),
            )

            #example_outputs[example_outputs == self.vocab.get_id(";")] = self.vocab.get_id(self.vocab.noop)

            examples[split] = (example_inputs, example_outputs)
            '''
            # origin
            
        self.dataset = {
            "train": SimpleDataset(
                examples=examples["train"], dfas=DFAs["train"], tokenizer=self.tokenizer
            ),
            "valid": SimpleDataset(
                examples=examples["valid"], dfas=DFAs["valid"], tokenizer=self.tokenizer
            ),
            "icl__": SimpleDataset(
                examples=examples["icl__"], dfas=DFAs["icl__"], tokenizer=self.tokenizer
            ),
            "ood__": SimpleDataset(
                examples=examples["ood__"], dfas=DFAs["icl__"], tokenizer=self.tokenizer
            ),
            "ood1_": SimpleDataset( # noisy input
                examples=examples["ood1_"], dfas=DFAs["icl__"], tokenizer=self.tokenizer
            ),
            "ood2_": SimpleDataset( # noisy input
                examples=examples["ood2_"], dfas=DFAs["icl__"], tokenizer=self.tokenizer
            ),
        }

    def _collate_fn(self, batch):
        icl_ks, nm_examples_x, nm_examples_y, tv_examples_xs, tv_examples_ys, dfas = zip(*batch)
        nm_xs = torch.stack(nm_examples_x)
        nm_ys = torch.stack(nm_examples_y)
        tv_xs = torch.cat(tv_examples_xs) 
        tv_ys = torch.cat(tv_examples_ys)
        icl_ks = icl_ks
        tv_dfas_index = []
        for idx, icl_k in enumerate(icl_ks):
            tv_dfas_index.extend([idx]*icl_k)
        return {'icl_ks': icl_ks,
                'tv_dfas_index': tv_dfas_index,
                'nm_xs': nm_xs, 
                'nm_ys': nm_ys, 
                'tv_xs': tv_xs, 
                'tv_ys': tv_ys,
                'dfas': dfas}

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["train"], shuffle=True)

    def valid_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["valid"], shuffle=False)

    def icl___dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["icl__"], shuffle=False)

    def ood___dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["ood__"], shuffle=False)
    
    def ood1__dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["ood1_"], shuffle=False)
    
    def ood2__dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["ood2_"], shuffle=False)
    
    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            persistent_workers=True,
        )


if __name__ == "__main__":
    # test dataloader
    data_module = ICLDFADataModule(
        num_dfas = 10,
        num_ood_dfas = 100,
        num_ood_samples = 3,
        num_random_labels = 3,
        insert_task_vector = 1,
        insert_mode = 'last',
        num_examples = 5000,
        num_valid_examples = 250,
        num_icl___examples = 250,
        vocab_size = 40,
        max_num_nodes = 12,
        max_num_in_context_examples = 11,
        min_num_in_context_examples = 10,
        max_outgoing_edges=4,
        max_len_per_example=11,
        min_len_per_example=10,
        number_duplicates_per_epoch=0,
        input_seq_len=4096*2,
        seed=0,
        batch_size=32,
        split_train_test=False,
        data_dir=None,
    )

    data_module.setup(num_icl_samples=10, icl_sample_length=9, mode = 'save')
    
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()
    icl___loader = data_module.icl___dataloader()
    ood___loader = data_module.ood___dataloader()
    ood1__loader = data_module.ood1__dataloader()
    ood2__loader = data_module.ood2__dataloader()

    for batch in tqdm(ood___loader):
        print(batch['nm_xs'][-1])
        
        break

    data_module.setup(num_icl_samples=10, icl_sample_length=9, mode = 'load')
    
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()
    icl___loader = data_module.icl___dataloader()
    ood___loader = data_module.ood___dataloader()

    for batch in tqdm(ood___loader):
        print(batch['nm_xs'][-1])
        
        break
    
    data_module.setup(num_icl_samples=10, icl_sample_length=9, mode = 'load')
    
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()
    icl___loader = data_module.icl___dataloader()
    ood___loader = data_module.ood___dataloader()

    for batch in tqdm(ood___loader):
        print(batch['nm_xs'][-1])
        
        break
    
    data_module.setup(num_icl_samples=10, icl_sample_length=9, mode = 'save')
    
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.valid_dataloader()
    icl___loader = data_module.icl___dataloader()
    ood___loader = data_module.ood___dataloader()

    for batch in tqdm(ood___loader):
        print(batch['nm_xs'][-1])
        
        break
