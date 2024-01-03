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
    生成一个dataloader用于训练
    """
    vocab = SELFIEVocab(vocab_path)

    #获得selfies表示
    dataset = SMILESDataset(dataset_dir, percentage, vocab)

    def pad_collate(batch):
        """
        通过填充(padding)，将不同长度的序列放入一个minibatch中
        """
        lengths = [len(x) for x in batch]

        # 嵌入层需要long张量
        batch = [torch.tensor(x, dtype=torch.long) for x in batch]

        #pad_sequence，input的tensor需要处理成定长的，例如[1],[1,2]->[1,0],[1,2]
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
        smiles_file: .smi 文件路径包含 SMILES.
        percantage: 要使用的数据库的百分比.
        """
        super(SMILESDataset, self).__init__()
        assert(0 < percentage <= 1)

        self.percentage = percentage
        self.vocab = vocab

        #read_smiles_file将Smile文件中的对应比例读成一个list，用data存放
        self.data = self.read_smiles_file(smiles_file)
        print("total number of SMILES loaded: ", len(self.data))

        # sf.encoder()将smile转化为selfies
        if self.vocab.name == "selfies":
            self.data = [sf.encoder(x)
                         for x in self.data if sf.encoder(x) is not None]
            print("有效的 SELFIES 个数为: ", len(self.data))

    def read_smiles_file(self, path):
        # 需要排除第一行，这一行不是SMILES
        with open(path, "r") as f:
            smiles = [line.strip("\n") for line in f.readlines()]

        num_data = len(smiles)

        return smiles[0:int(num_data * self.percentage)]

    def __getitem__(self, index):
        mol = self.data[index]

        # 将数据转化为整数tokens
        mol = self.vocab.tokenize_smiles(mol)

        return mol

    def __len__(self):
        return len(self.data)

class SELFIEVocab:
    def __init__(self, vocab_path):
        self.name = "selfies"

        # 导入提前计算好的词汇表
        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        self.int2tocken = {value: key for key, value in self.vocab.items()}

    def tokenize_smiles(self, mol):
        """将smiles 转化为 selfies, 然后返回整数 tokens."""
        ints = [self.vocab['<sos>']]

        #encoded_selfies = sf.encoder(smiles)
        selfies_list = list(sf.split_selfies(mol))
        for token in selfies_list:
            ints.append(self.vocab[token])

        ints.append(self.vocab['<eos>'])

        return ints

    def combine_list(self, selfies):
        return "".join(selfies)
