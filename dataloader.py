# Copyright: Wentao Shi, 2021
import torch
import re
import yaml
import selfies as sf

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def dataloader_gen(dataset_dir, percentage, which_vocab, vocab_path,
                   batch_size, PADDING_IDX, shuffle, drop_last=True):
    """
    Genrate the dataloader for training
    """
    vocab = SELFIEVocab(vocab_path)


    dataset = SMILESDataset(dataset_dir, percentage, vocab)

    def pad_collate(batch):
        """
        Put the sequences of different lengths in a minibatch by paddding.
        """
        lengths = [len(x) for x in batch]

        # embedding layer takes long tensors
        batch = [torch.tensor(x, dtype=torch.long) for x in batch]

        x_padded = pad_sequence(
            batch, 
            batch_first=True,
            padding_value=PADDING_IDX
        )

        return x_padded, lengths

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last, 
        collate_fn=pad_collate
    )

    return dataloader, len(dataset)


class SMILESDataset(Dataset):
    def __init__(self, smiles_file, percentage, vocab):
        """
        smiles_file: path to the .smi file containing SMILES.
        percantage: percentage of the dataset to use.
        """
        super(SMILESDataset, self).__init__()
        assert(0 < percentage <= 1)

        self.percentage = percentage
        self.vocab = vocab

        # load eaqual portion of data from each tranche
        self.data = self.read_smiles_file(smiles_file)
        print("total number of SMILES loaded: ", len(self.data))

        # convert the smiles to selfies
        if self.vocab.name == "selfies":
            self.data = [sf.encoder(x)
                         for x in self.data if sf.encoder(x) is not None]
            print("total number of valid SELFIES: ", len(self.data))

    def read_smiles_file(self, path):
        # need to exclude first line which is not SMILES
        with open(path, "r") as f:
            smiles = [line.strip("\n") for line in f.readlines()]

        num_data = len(smiles)

        return smiles[0:int(num_data * self.percentage)]

    def __getitem__(self, index):
        mol = self.data[index]

        # convert the data into integer tokens
        mol = self.vocab.tokenize_smiles(mol)

        return mol

    def __len__(self):
        return len(self.data)

class SELFIEVocab:
    def __init__(self, vocab_path):
        self.name = "selfies"

        # load the pre-computed vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        self.int2tocken = {value: key for key, value in self.vocab.items()}

    def tokenize_smiles(self, mol):
        """convert the smiles to selfies, then return 
        integer tokens."""
        ints = [self.vocab['<sos>']]

        #encoded_selfies = sf.encoder(smiles)
        selfies_list = list(sf.split_selfies(mol))
        for token in selfies_list:
            ints.append(self.vocab[token])

        ints.append(self.vocab['<eos>'])

        return ints

    def combine_list(self, selfies):
        return "".join(selfies)
