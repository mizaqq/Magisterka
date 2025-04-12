import json
from pathlib import Path

import hydra

from src.model.classifier import XGBoost
from src.model.dataloader import Dataloader


@hydra.main(config_path="./conf", config_name="config")
def main(cfg) -> None:
    dataloader = Dataloader()
    dataloader.get_encoder()
    train_batch, test_batch = dataloader.get_training_ready_data(0.3)
    model = XGBoost(**cfg.params)
    model.fit(train_batch['embeddings'], train_batch['Category'])
    results = model.test_model(test_batch['embeddings'], test_batch['Category'])
    # print_metrics(results=results)
    model.save_model('/home/miza/Magisterka/src/model/xgboost_model.json')

    hydra.utils.log.info("Training completed successfully.")
    gpt_data = dataloader.get_gpt_data_for_finetune()
    model.fit(gpt_data['embeddings'], gpt_data['Category'])
    test_results = model.test_model(test_batch['embeddings'], test_batch['Category'])
    # print_metrics(test_results)
    print("Test results:{}, {}".format(results['f1_score'], test_results['f1_score']))
    return (results['f1_score'] + test_results['f1_score']) / 2


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
