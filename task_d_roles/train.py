#!/usr/bin/env  python3
# *-* coding: utf-8 *-*

import enum
from hyper_parameters import HyperParameters
import numpy
import pandas
import sklearn.model_selection
import os
import random
import sys
import torch
import transformers
import typing
import matplotlib

ROOT: typing.Final[str] = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT))

import src.trainer
import src.dataset
import src.metrics
import src.model
import src.data_utils


# Hyperparameters

DATA_FILE: typing.Final[str] = "../HopeEXP_Train.jsonl"
SEED: typing.Final[int] = 42
OUTPUT_DIR: typing.Final[str] = "output"
TOKEN_SAMPLE_SIZE: typing.Final[int] = 500
NUM_WORKERS: typing.Final[int] = 12


def set_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Colour(enum.Enum):
    Black = 0
    Red = 1
    Green = 2
    Yellow = 3
    Blue = 4
    Magenta = 5
    Cyan = 6
    White = 7

    def __str__(self) -> str:
        if self == Colour.White:
            return "\033[38m"
        else:
            return f"\033[3{self.value}m"


class Format(enum.Enum):
    Clear = 0
    Bold = 1
    def __str__(self) -> str:
        match self:
            case Format.Clear:
                return "\033[0m"
            case Format.Bold:
                return "\033[1m"


class Log:
    def __init__(self, indent: int):
        self.__indent_size : typing.Final[int] = indent
        self.__level = 0

    def indent(self):
        self.__level += 1

    def deindent(self):
        self.__level = max(0, self.__level - 1)

    def put(self, *args, **kwargs):
        print(self.__level * self.__indent_size * ' ', **kwargs, end="")
        for arg in args:
            if isinstance(arg, Colour) or isinstance(arg, Format):
                print(arg, **kwargs, end="")
            else:
                print(arg, **kwargs, end=" ")
        print(**kwargs)


log = Log(3)


class OutcomeStance(enum.Enum):
    Avoided = 0
    Desired = 1

    def __lt__(self, other: typing.Self) -> bool:
        return self.value < other.value


class Actor(enum.Enum):
    Self = 0
    Other = 1
    World = 2
    Unclear = 3

    def __lt__(self, other: typing.Self) -> bool:
        return self.value < other.value


class Language(enum.Enum):
    ES = 0
    EN = 1

    def __lt__(self, other: typing.Self) -> bool:
        return self.value < other.value


def encode(label: enum.Enum) -> list[int]:
    # Encode as booleans
    result = numpy.array([0] * len(type(label)), dtype="?")
    result[label.value] = True
    return result


def to_actor(x: str) -> Actor:
    return Actor.World if x == "World/System" else Actor[x]


def load_data() -> tuple[pandas.DataFrame, pandas.DataFrame]:
    all_data = pandas.read_json(path_or_buf=DATA_FILE, lines=True)
    lang_row = list(all_data.keys()).index("lang")
    span_annotations_row = list(all_data.keys()).index("span_annotations")
    # For outcomes
    outcome_stance = pandas.DataFrame({
        "span": [item["span"]
                 for items in all_data["span_annotations"]
                 for item in items],
        "lang": [Language[row[lang_row]]
                 for row in all_data.itertuples(index=False)
                 for item in row[span_annotations_row]],
        "label": [OutcomeStance[item["outcome_stance"]]
                  for items in all_data["span_annotations"]
                  for item in items]})
    # TODO: Maybe encode it just like a boolean.
    outcome_stance["enc"] = outcome_stance["label"].apply(encode)
    # For actors
    actor = pandas.DataFrame({
        "span": [item["span"]
                 for items in all_data["span_annotations"]
                 for item in items],
        "lang": [Language[row[lang_row]]
                 for row in all_data.itertuples(index=False)
                 for item in row[span_annotations_row]],
        "label": [to_actor(item["actor"])
                  for items in all_data["span_annotations"]
                  for item in items]})
    actor["enc"] = actor["label"].apply(encode)
    return outcome_stance, actor


def show_distribution(data, train, dev):
    log.put(Format.Bold, "Data set:", Format.Clear, len(data), "samples")
    log.put(Format.Bold, "Train:   ", Format.Clear, len(train), "samples")
    log.put(Format.Bold, "Dev:     ", Format.Clear, len(dev), "samples")
    log.put(Format.Bold, "Label distribution", Format.Clear)
    data_dist = data.groupby("label").size().rename("total")
    train_dist = train.groupby("label").size().rename("train")
    dev_dist = dev.groupby("label").size().rename("dev")
    print(pandas.concat([data_dist, train_dist, dev_dist], axis=1))

def train(data: pandas.DataFrame, params: HyperParameters, device):
    # Split in train and test
    train, dev = sklearn.model_selection.train_test_split(
        data, test_size=params.dev_size, random_state=SEED,
        stratify=data["label"])
    train = train.reset_index(drop=True)
    dev = dev.reset_index(drop=True)
    show_distribution(data, train, dev)
    labels = list(type(data["label"][0]))
    pos_weights = src.data_utils.compute_class_weights(train, "enc", labels)
    pos_weights = torch.tensor(pos_weights) if params.use_pos_weight else None
    log.indent()

    # Tokenise
    tokeniser = transformers.AutoTokenizer.from_pretrained(params.model_name)
    log.put(Colour.Yellow, "Tokeniser loaded!", Format.Clear)
    lengths = [len(tokeniser.encode(span, add_special_tokens=True))
               for span in train["span"][:TOKEN_SAMPLE_SIZE]]
    log.put(Format.Bold, "Sample size:               ", Format.Clear,
            TOKEN_SAMPLE_SIZE)
    log.put(Format.Bold, "Mean token length:         ", Format.Clear,
            f"{numpy.mean(lengths):.2f}")
    log.put(Format.Bold, "95th percentil:            ", Format.Clear,
            f"{int(numpy.percentile(lengths, 95))}")
    log.put(Format.Bold, "Maximum length:            ", Format.Clear,
            f"{max(lengths)}")
    log.put(Format.Bold, "Configured maximum length: ", Format.Clear,
            params.max_length)
    loaders = src.dataset.build_all_dataloaders(
            train_texts=train["span"],
            train_labels=train["enc"],
            dev_texts=dev["span"],
            dev_labels=dev["enc"],
            tokenizer=tokeniser,
            test_texts=None,
            max_length=params.max_length,
            batch_size=params.batch_size,
            num_workers=NUM_WORKERS)

    # Generate the model
    log.put(Colour.Yellow, "Building model", Format.Clear)
    model = src.model.build_model(
            model_name=params.model_name,
            num_labels=params.num_labels,
            dropout_prob=params.dropout).float()
    n_params = src.model.count_parameters(model)
    log.put(Format.Bold, "Trainable parameters:", Format.Clear, n_params)
    log.put(Format.Bold, "Classification head: ", Format.Clear,
            params.num_labels, "labels")

    # Training
    log.put(Colour.Yellow, "Building model", Format.Clear)
    os.makedirs(params.save_dir, exist_ok=True)
    history = src.trainer.train(
            model=model,
            train_loader=loaders["train"],
            dev_loader=loaders["dev"],
            device=device,
            num_epochs=params.num_epochs,
            learning_rate=params.lr,
            warmup_ratio=params.warmup_ration,
            weight_decay=params.weight_decay,
            threshold=params.threshold,
            pos_weight=pos_weights,
            save_dir=params.save_dir,
            model_name=params.best_model_name,
            early_stopping_patience=params.patience,
            monitor_metric="f1_macro",
            verbose=True)


    log.deindent()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.put(Format.Bold, "Device:", Format.Clear, device)
    if device.type == "cuda":
        log.put(Format.Bold, "GPU:   ", Format.Clear,
                torch.cuda.get_device_name(0))

    outcome_stance_data, actor_data = load_data()
    outcome_stance_params = HyperParameters(
        save_dir="./outcome_stance_output",
        num_labels=len(OutcomeStance))
    actor_params = HyperParameters(
        save_dir="./actor_output",
        num_labels=len(Actor))

    log.put(Format.Bold, Colour.Yellow,
            "==== Training Outcome Stance Model ====", Format.Clear)
    train(outcome_stance_data, outcome_stance_params, device)

    log.put()
    log.put(Format.Bold, Colour.Yellow,
            "==== Training Actor Model ====", Format.Clear)
    train(actor_data, actor_params, device)


if __name__ == "__main__":
    main()
