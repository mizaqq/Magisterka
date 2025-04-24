import json
from pathlib import Path

import hydra

from src.model.classifier import XGBoost
from src.model.dataloader import Dataloader
from src.model.embeddings import Embedding


@hydra.main(config_path="./conf", config_name="config")
def main(cfg) -> None:
    dataloader = Dataloader(Path(cfg.datapath))
    dataloader.get_encoder()
    if cfg.embeddingmodel == 'bert':
        train_batch, test_batch = dataloader.get_training_ready_data(0.3)
        gpt_data = dataloader.get_gpt_data_for_finetune(Path(cfg.gptdatapath))

    elif cfg.embeddingmodel == 'mini':
        data = dataloader.get_preprocessed_data()
        train_data, test_data = dataloader.split_data(data, 0.3)
        if cfg.finetune:
            Embedding.finetune_mini(train_data)
        embedding_model = Embedding.load_mini_model()
        train_embeddings = embedding_model.encode(train_data['text'].tolist())
        test_embeddings = embedding_model.encode(test_data['text'].tolist())
        train_batch = {
            'embeddings': train_embeddings,
            'category': train_data['category'],
        }
        test_batch = {
            'embeddings': test_embeddings,
            'category': test_data['category'],
        }
        gpt_data = dataloader.get_gpt_data_for_finetune_with_balance_and_mini(embedding_model, Path(cfg.gptdatapath))
    model = XGBoost(**cfg.params)
    model.fit(train_batch['embeddings'], train_batch['category'])
    results = model.test_model(test_batch['embeddings'], test_batch['category'])
    print_metrics(results=results)
    model.save_model('/home/miza/Magisterka/src/model/xgboost_model.json')
    model.fit(gpt_data['embeddings'], gpt_data['category'])
    test_results = model.test_model(test_batch['embeddings'], test_batch['category'])
    print_metrics(test_results)
    print("Test results:{}, {}".format(results['f1_score'], test_results['f1_score']))
    hydra.utils.log.info(f'Test results: {(results["f1_score"] + test_results["f1_score"]) / 2}')
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
