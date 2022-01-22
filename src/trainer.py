from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from transformers import DistilBertTokenizer

import training_utils
import visualization
from datasets import NewsDataset
from models import DistilBERT


class ClassificationTrainer:

    def __init__(self, model_parameters, training_parameters):

        self.model_parameters = model_parameters
        self.training_parameters = training_parameters

    def train_fn(self, train_loader, model, criterion, optimizer, device, scheduler=None):

        """
        Train given model on given data loader

        Parameters
        ----------
        train_loader (torch.utils.data.DataLoader): Training set data loader
        model (torch.nn.Module): Model to train
        criterion (torch.nn.modules.loss): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Location of the model and inputs
        scheduler (torch.optim.LRScheduler or None): Learning rate scheduler

        Returns
        -------
        train_loss (float): Average training loss after model is fully trained on training set data loader
        """

        print('\n')
        model.train()
        progress_bar = tqdm(train_loader)
        losses = []

        for sequences, label in progress_bar:
            input_ids, token_type_ids, attention_mask, label = sequences['input_ids'].to(device), sequences['token_type_ids'].to(device), sequences['attention_mask'].to(device), label.to(device)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(label, output)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
            average_loss = np.mean(losses)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            progress_bar.set_description(f'train_loss: {average_loss:.6f} - lr: {lr:.8f}')

        train_loss = np.mean(losses)
        return train_loss

    def val_fn(self, val_loader, model, criterion, device):

        """
        Validate given model on given data loader

        Parameters
        ----------
        val_loader (torch.utils.data.DataLoader): Validation set data loader
        model (torch.nn.Module): Model to validate
        criterion (torch.nn.modules.loss): Loss function
        device (torch.device): Location of the model and inputs

        Returns
        -------
        val_loss (float): Average validation loss after model is fully validated on validation set data loader
        """

        model.eval()
        progress_bar = tqdm(val_loader)
        losses = []
        targets = []
        predictions = []

        with torch.no_grad():
            for sequences, label in progress_bar:
                input_ids, token_type_ids, attention_mask, target = sequences['input_ids'].to(device), sequences['token_type_ids'].to(device), sequences['attention_mask'].to(device), label.to(device)
                output = model(input_ids, attention_mask)
                loss = criterion(output, target)
                losses.append(loss.item())
                average_loss = np.mean(losses)
                progress_bar.set_description(f'val_loss: {average_loss:.6f}')
                targets += target.detach().cpu().numpy().tolist()
                predictions += output.detach().cpu().numpy().tolist()

        val_loss = np.mean(losses)
        return val_loss

    def train_and_validate(self, df):

        """
        Train and validate on texts and labels listed on given dataframe with specified configuration

        Parameters
        ----------
        df [pandas.DataFrame of shape (n_samples, 7)]: Dataframe of texts, labels and folds
        """

        print(f'\n{"-" * 30}\nRunning Model for Training\n{"-" * 30}\n')

        trn_idx, val_idx = df.loc[df['fold'] == 'train'].index, df.loc[df['fold'] == 'val'].index
        train_dataset = NewsDataset(
            df.loc[trn_idx, 'content'].values,
            df.loc[trn_idx, self.model_parameters['labels']].values,
            tokenizer=DistilBertTokenizer.from_pretrained(self.model_parameters['model_class_parameters']['pretrained_model_path']),
            max_seq_len=self.model_parameters['max_seq_len']
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_parameters['data_loader']['training_batch_size'],
            sampler=RandomSampler(train_dataset),
            pin_memory=True,
            drop_last=False,
            num_workers=self.training_parameters['data_loader']['num_workers']
        )
        val_dataset = NewsDataset(
            df.loc[val_idx, 'content'].values,
            df.loc[trn_idx, self.model_parameters['labels']].values,
            tokenizer=DistilBertTokenizer.from_pretrained(self.model_parameters['model_class_parameters']['pretrained_model_path']),
            max_seq_len=self.model_parameters['max_seq_len']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_parameters['data_loader']['test_batch_size'],
            sampler=SequentialSampler(val_dataset),
            pin_memory=True,
            drop_last=False,
            num_workers=self.training_parameters['data_loader']['num_workers']
        )

        training_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
        device = torch.device(self.training_parameters['device'])
        criterion = nn.CrossEntropyLoss()
        model = DistilBERT(**self.model_parameters['model_class_parameters'])
        if self.model_parameters['model_checkpoint_path'] is not None:
            model_checkpoint_path = self.model_parameters['model_checkpoint_path']
            model.load_state_dict(torch.load(model_checkpoint_path))
        model.to(device)

        optimizer = getattr(optim, self.training_parameters['optimizer'])(model.parameters(), **self.training_parameters['optimizer_parameters'])
        scheduler = getattr(optim.lr_scheduler, self.training_parameters['lr_scheduler'])(optimizer, **self.training_parameters['lr_scheduler_parameters'])

        early_stopping = False
        summary = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(1, self.training_parameters['epochs'] + 1):

            if early_stopping:
                break

            if self.training_parameters['lr_scheduler'] == 'ReduceLROnPlateau':
                train_loss = self.train_fn(train_loader, model, criterion, optimizer, device, scheduler=None)
                val_loss = self.val_fn(val_loader, model, criterion, device)
                scheduler.step(val_loss)
            else:
                train_loss = self.train_fn(train_loader, model, criterion, optimizer, device, scheduler)
                val_loss = self.val_fn(val_loader, model, criterion, device)

            print(f'Epoch {epoch} - Training Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f}')
            best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
            if val_loss < best_val_loss:
                model_path = f'{self.model_parameters["model_path"]}/{self.model_parameters["model_filename"]}.pt'
                torch.save(model.state_dict(), model_path)
                print(f'Saving model to {model_path} (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})')

            summary['train_loss'].append(train_loss)
            summary['val_loss'].append(val_loss)

            best_iteration = np.argmin(summary['val_loss']) + 1
            if len(summary['val_loss']) - best_iteration >= self.training_parameters['early_stopping_patience']:
                print(f'Early stopping (validation loss didn\'t increase for {self.training_parameters["early_stopping_patience"]} epochs/steps)')
                print(f'Best validation loss is {np.min(summary["val_loss"]):.6f}')
                visualization.visualize_learning_curve(
                    training_losses=summary['train_loss'],
                    validation_losses=summary['val_loss'],
                    title=f'{self.model_parameters["model_filename"]} - Learning Curve',
                    path=f'{self.model_parameters["model_path"]}/{self.model_parameters["model_filename"]}_learning_curve.png'
                )
                early_stopping = True
