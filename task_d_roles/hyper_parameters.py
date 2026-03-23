# *-* coding: utf-8 *-*

import dataclasses
import typing

@dataclasses.dataclass
class HyperParameters:
    save_dir: typing.Final[str]
    num_labels: typing.Final[int]
    dev_size: typing.Final[float] = 0.20
    best_model_name: typing.Final[str] = "best_model.pt"

    # Training
    batch_size: typing.Final[int] = 4
    num_epochs: typing.Final[int] = 10
    lr: typing.Final[float] = 2e-5
    warmup_ration: typing.Final[float] = 0.1
    weight_decay: typing.Final[float] = 0.01
    patience: typing.Final[int] = 3

    # Model
    model_name: typing.Final[str] = "microsoft/mdeberta-v3-base"
    max_length: typing.Final[int] = 48
    dropout: typing.Final[float] = 0.1

    #  Evaluation
    threshold: typing.Final[float] = 0.5
    use_pos_weight: typing.Final[bool] = True
