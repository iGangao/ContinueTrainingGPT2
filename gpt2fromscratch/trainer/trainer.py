import torch
import torch.nn as nn
import logging
from ..utils.gpt2dataset import GPT2Dataset
from ..model.gpt2 import GPT2
from ..configs.trainerconfig import TrainerConfig
# from GPT2Trainer.dataset import CustomDataset
logger = logging.getLogger(__name__)
class Trainer:
    def __init__(self, model: GPT2, dataset: GPT2Dataset, config: TrainerConfig):

        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cpu')

        if self.config.use_gpu:
            assert torch.cuda.is_available(), "No GPU Found"
            self.device = torch.device('cuda')

        # Data loader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate)

        # Wrap the model with FSDP if configured
        if self.config.use_fsdp:
            self.model = torch.distributed.fsdp.FullyShardedDataParallel(self.model)

        # Distributed setup if using DDP
        if self.config.use_ddp:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.model = nn.parallel.DistributedDataParallel(self.model)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"saving {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        for epoch in range(self.config.num_epochs):
            for batch in self.dataloader:
                input_sequence, target_sequence = batch
                # Move data to device
                input_sequence, target_sequence = input_sequence.to(self.device), target_sequence.to(self.device)

                # Forward pass
                output = self.model(input_sequence)
                output_logits = output[:, -1, :]

                # Compute loss
                loss = self.criterion(output_logits, target_sequence)
                self.optimizer.zero_grad()
                loss.backward()

                # Update parameters using the optimizer
                with torch.no_grad():
                    self.optimizer.step()

                # Print statistics
                print(f'Epoch: {epoch + 1}/{self.config.num_epochs}, Loss: {loss.item()}')
