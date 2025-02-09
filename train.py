# Copyright: Wentao Shi, 2021
import yaml
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from rdkit import Chem
import selfies as sf

from dataloader import dataloader_gen
from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN

# suppress rdkit error
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def make_vocab(config):
    # 加载 vocab
    which_vocab = config["which_vocab"]
    vocab_path = config["vocab_path"]

    return SELFIEVocab(vocab_path)
    

def sample(model, vocab, batch_size):
    """从当前模型 采样 a batch of SMILES ."""
    model.eval()
    # 采样
    sampled_ints = model.sample(
        batch_size=batch_size,
        vocab=vocab,
        device=device
    )

    # 整数tokens 转换为 SELFIES
    molecules = []
    sampled_ints = sampled_ints.tolist()
    for ints in sampled_ints:
        molecule = []
        for x in ints:
            if vocab.int2tocken[x] == '<eos>':
                break
            else:
                molecule.append(vocab.int2tocken[x])
        molecules.append("".join(molecule))

    # SELFIES 转换回SMILES
    if vocab.name == 'selfies':
        molecules = [sf.decoder(x) for x in molecules]

    return molecules


def compute_valid_rate(molecules):
    """计算有效 SMILES 的百分比 给定一个SMILES字符串的列表"""
    num_valid, num_invalid = 0, 0
    for mol in molecules:
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            num_invalid += 1
        else:
            num_valid += 1

    return num_valid, num_invalid


if __name__ == "__main__":
    # 选择 cpu / gpu
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print('device: ', device)

    config_dir = "./train.yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)

    # 输出结果的目录
    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trained_model_dir = out_dir + 'trained_model.pt'

    # 保存配置文件，以待未来使用
    with open(out_dir + 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # 训练数据
    dataset_dir = config['dataset_dir']
    which_vocab = config['which_vocab']
    vocab_path = config['vocab_path']
    percentage = config['percentage']

    # 创建 dataloader
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    PADDING_IDX = config['rnn_config']['num_embeddings'] - 1
    num_workers = os.cpu_count()
    print('加载数据的CPU核心个数: ', num_workers)
    print('使用的词汇表: ', which_vocab)
    dataloader, train_size = dataloader_gen(
        dataset_dir, percentage, which_vocab,
        vocab_path, batch_size, PADDING_IDX,
        shuffle, drop_last=False
    )

    # 模型和训练参数配置
    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']

    # reduction="sum" 让输出loss求和，会让采样分子的有效率有很大区别
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay, amsgrad=True
    )

    # 学习率 scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=5,
        cooldown=10, min_lr=0.0001,
        verbose=True
    )

    # sample() 函数使用的词汇对象
    vocab = make_vocab(config)

    # 训练和验证, 结果会被保存.
    train_losses = []
    best_valid_rate = 0
    num_epoch = config['num_epoch']

    print('开始训练...')
    for epoch in range(1, 1 + num_epoch):
        model.train()
        train_loss = 0
        for data, lengths in tqdm(dataloader):
            # 长度-1，是因为我们在训练时，输入时不用<eos>，输出不需要<sos>
            
            lengths = [length - 1 for length in lengths]

            optimizer.zero_grad()
            data = data.to(device)
            preds = model(data, lengths)

            # 在packing之前，删除<sos> token, 因为
            # 在训练时我们不需要输出的<sos>.
            # the image_captioning 项目使用相同的方法
            # 它直接把packed的序列喂给
            # 损失函数:
            # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/train.py
            targets = pack_padded_sequence(
                data[:, 1:],
                lengths,
                batch_first=True,
                enforce_sorted=False
            ).data

            loss = loss_function(preds, targets)
            loss.backward()
            optimizer.step()

            # 累加小批量的loss
            train_loss += loss.item()  # * data.size()[0]

        train_losses.append(train_loss / train_size)

        print('周期 {}, 训练损失: {}.'.format(epoch, train_losses[-1]))

        scheduler.step(train_losses[-1])

        # 每个周期采样1024个SMILE
        sampled_molecules = sample(model, vocab, batch_size=1024)

        # 每个周期输出有效率(valid_rate)
        num_valid, num_invalid = compute_valid_rate(sampled_molecules)
        valid_rate = num_valid / (num_valid + num_invalid)

        print('有效率: {}'.format(valid_rate))

        # 更新已保存的模型，当有效率高于当前最佳有效率时
        if valid_rate >= best_valid_rate:
            best_valid_rate = valid_rate
            print('model saved at epoch {}'.format(epoch))
            torch.save(model.state_dict(), trained_model_dir)

    # 保存训练和验证的损失
    with open(out_dir + 'loss.yaml', 'w') as f:
        yaml.dump(train_losses, f)
