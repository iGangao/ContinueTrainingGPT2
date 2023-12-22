from dataclasses import dataclass
@dataclass
class TrainerConfig:
    """
    Configuration class for the model trainer.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training.
        num_epochs (int): The number of training epochs.
        use_ddp (bool): Whether to use Distributed Data Parallel (DDP) for training.
        use_fsdp (bool): Whether to use Fully Sharded Data Parallelism (FSDP) for training.
        use_gpu (bool): Whether to use GPU for training.
    """

    learning_rate: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 1
    use_ddp: bool = False
    use_fsdp: bool = False
    use_gpu: bool = True