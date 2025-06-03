import json
from pathlib import Path

import hydra

from src.model.classifier import XGBoost
from src.model.dataloader import Dataloader
from src.model.embeddings import Embedding, BertModelWrapper
import numpy as np
import random
import pandas as pd


@hydra.main(config_path="./conf", config_name="config")
def main(cfg) -> None:
    dataloader = Dataloader(Path(cfg.datapath))
    dataloader.get_encoder()
    if cfg.finetune.finetune:
        data = dataloader.get_preprocessed_data()
        train_data, _, val_data = dataloader.split_data(data, 0.25)
        gpt_data = dataloader.get_gpt_data_for_training(Path(cfg.gptdatapath))
        Embedding.finetune_mini(train_data, val_data, path=cfg.finetune.model_base)
        data_together = pd.concat([train_data, gpt_data], ignore_index=True)
        Embedding.finetune_mini(data_together, val_data, path=cfg.finetune.model_augmented)
    if 'bert' in cfg.embeddingmodel.model_path:
        embedding_model = BertModelWrapper()
    elif 'mini' in cfg.embeddingmodel.model_path:
        embedding_model = Embedding.load_mini_model(cfg.embeddingmodel.model_path)
    train_batch, test_batch, validation_batch = dataloader.get_training_ready_data(
        embedding_model, split_size=cfg.split_size
    )
    gpt_data = dataloader.get_gpt_data_training_ready_data(embedding_model, Path(cfg.gptdatapath), cfg.balance_data)
    joined_data = dataloader.get_joined_data(
        train_batch['embeddings'], train_batch['category'], gpt_data['embeddings'], gpt_data['category']
    )

    model = XGBoost(**cfg.params)
    model.fit(train_batch['embeddings'], train_batch['category'])
    results = model.test_model(test_batch['embeddings'], test_batch['category'])
    print_metrics(results=results)
    print("Classes: ", [f"{i}:{c}" for i, c in enumerate(dataloader.le.classes_)])

    model.fit(joined_data['embeddings'], joined_data['category'])
    test_results = model.test_model(test_batch['embeddings'], test_batch['category'])
    print_metrics(test_results)
    print("Classes: ", [f"{i}:{c}" for i, c in enumerate(dataloader.le.classes_)])

    hydra.utils.log.info(f'Test results: {results["f1_score"]}')
    hydra.utils.log.info(f'Augumented Test results:{test_results["f1_score"]}')
    return max(results['f1_score'], test_results['f1_score'])


def print_metrics(
    results: dict,
) -> None:
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print("Classification Report:")
    print(results['classification_report'])


def save_metrics(
    results: dict,
) -> None:
    output_dir = Path.cwd()
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
