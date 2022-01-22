from torch.utils.data import Dataset
import torch


class NewsDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_seq_len):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def encode(self, text, max_seq_len):

        """
        Tokenize given text into a fixed length sequence

        Parameters
        ----------
        text (str): Plain text string
        max_seq_len (str): Maximum number of tokens in the sequence

        Returns
        -------
        sequences (dict): Dictionary of sequences
        Dictionary contains:
            - input_ids [list of shape (self.max_seq_len)]: Tokens
            - token_type_ids [list of shape (self.max_seq_len)]: Token types
            - attention_mask [list of shape (self.max_seq_len)]: Attention mask
        """

        text = text.replace('\n', '')
        tokenized = self.tokenizer.encode_plus(
            text,
            max_length=max_seq_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )

        sequences = {}
        # Pad zeros at the end of sequence until sequence length reaches max_seq_len
        pad_len = max_seq_len - len(tokenized['input_ids'])
        sequences['input_ids'] = tokenized['input_ids'] + ([0] * pad_len)
        sequences['token_type_ids'] = tokenized['token_type_ids'] + ([0] * pad_len)
        sequences['attention_mask'] = tokenized['attention_mask'] + ([0] * pad_len)

        return sequences

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx (int): Index of the sample (0 <= idx < len(self.texts))

        Returns
        -------
        sequences (dict): Dictionary of sequences
        Dictionary contains:
            - input_ids [torch.Int64Tensor of shape (self.max_seq_len)]: Tokens
            - token_type_ids [torch.Int64Tensor of shape (self.max_seq_len)]: Token types
            - attention_mask [torch.Int64Tensor of shape (self.max_seq_len)]: Attention mask
        label [torch.FloatTensor of shape (1)]: Label
        """

        text, label = self.texts[idx], self.labels[idx]
        sequences = self.encode(text, self.max_seq_len)
        sequences = {
            'input_ids': torch.tensor(sequences['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(sequences['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sequences['attention_mask'], dtype=torch.long),
        }

        if self.labels is not None:
            label = torch.tensor(label, dtype=torch.float)
            return sequences, label
        else:
            return sequences
