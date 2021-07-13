from torchtext.legacy import data

class DataLoader(object):
    def __init__(
        self,
        train_fn,
        batch_size=128,
        valid_ratio=.1,
        device=-1,
        max_vocab=999999,
        min_freq=1,
        use_eos=True,
        shuffle=True,
    ):
        super().__init__()

        self.target = data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None
        )

        self.context = data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
            eos_token='<EOS>' if use_eos else None,
        )

        self.response = data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
            eos_token='<EOS>' if use_eos else None,
            fix_length = 256
        )

        train, valid = data.TabularDataset(
            path=train_fn,
            format='tsv',
            fields=[
                ('target', self.target),
                ('context', self.context),
                ('response', self.response)
            ],
        ).split(split_ratio=(1-valid_ratio))


        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' %device if device>=0 else 'cpu',
            shuffle=shuffle,
            sort_key=lambda x: len(x.context),
            sort_within_batch=True,
        )

        self.target.build_vocab(train)
        self.response.build_vocab(train, max_size=max_vocab, min_freq=min_freq)
        self.context.build_vocab(train, max_size=len(self.response.vocab), min_freq=min_freq)
        
