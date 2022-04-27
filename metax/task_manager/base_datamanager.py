import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


class MyQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False, uuids=None):
        self.uuids = uuids
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids) == len(
            self.attention_mask) == self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids) == len(
            self.decoder_attention_mask) == self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            in_idx = self.in_metadata[idx][0]
            out_idx = self.out_metadata[idx][0]
            # return self.input_ids[idx], self.attention_mask[idx]
            return self.input_ids[in_idx], self.attention_mask[in_idx], \
                self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        # TODO: can we pass the self.uuids[in_idx] ?
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler = RandomSampler(dataset)
            # args.train_batch_size is the batch size PER DEVICE
            batch_size = args.train_batch_size * args.n_gpu if args.n_gpu > 0 else args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(
            dataset, sampler=sampler, batch_size=batch_size, num_workers=1)
