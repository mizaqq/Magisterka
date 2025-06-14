---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1222
- loss:ContrastiveLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: OBNIZKA
  sentences:
  - SIATKAD PIZZY CM
  - OBNIZKA
  - TOPSPZYND SPR ZIM
- source_sentence: APRYKA CZERWONAKG
  sentences:
  - CYLINDER CO ROZ KAUCJA
  - SOK TtOCZ
  - SEL KIEEBASA SWOJSKA OK KG
- source_sentence: KOLAGEN EXTRA MAL TRUS
  sentences:
  - CYLINDER CO ROZ KAUCJA
  - KREM MPR
  - TOPSPZYND SPR ZIM
- source_sentence: ektar cz porzeczka
  sentences:
  - lecz dok zatr ro war
  - ALE PITNY  szt
  - SIATKAD PIZZY CM
- source_sentence: PRZYPRAHA GB
  sentences:
  - Chipsy Tortilla xxl
  - PUDLISZKI KETCHUP OGNISTY
  - Croissant maslany
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: dev
      type: dev
    metrics:
    - type: pearson_cosine
      value: 0.5747381074381046
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.5594124106083843
      name: Spearman Cosine
    - type: pearson_cosine
      value: 0.9276510293038853
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.8456769916124774
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'PRZYPRAHA GB',
    'PUDLISZKI KETCHUP OGNISTY',
    'Chipsy Tortilla xxl',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.5747     |
| **spearman_cosine** | **0.5594** |

#### Semantic Similarity

* Dataset: `dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.9277     |
| **spearman_cosine** | **0.8457** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,222 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                       | label                                                          |
  |:--------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                           | string                                                                           | float                                                          |
  | details | <ul><li>min: 3 tokens</li><li>mean: 7.77 tokens</li><li>max: 15 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 7.46 tokens</li><li>max: 15 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.57</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                  | sentence_1                       | label            |
  |:----------------------------|:---------------------------------|:-----------------|
  | <code>ODSWIEZACZ POW</code> | <code>ODPLAMIACZ KG</code>       | <code>1.0</code> |
  | <code>CUKINIA KG</code>     | <code>WYCISKACZ CZOSNKU</code>   | <code>0.0</code> |
  | <code>TASMA KLEJACA</code>  | <code>TOPS PLYN DOSPR ZIM</code> | <code>0.0</code> |
* Loss: [<code>ContrastiveLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#contrastiveloss) with these parameters:
  ```json
  {
      "distance_metric": "SiameseDistanceMetric.COSINE_DISTANCE",
      "margin": 0.5,
      "size_average": true
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 50
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 50
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch   | Step | Training Loss | dev_spearman_cosine |
|:-------:|:----:|:-------------:|:-------------------:|
| 1.0     | 77   | -             | -0.2001             |
| 2.0     | 154  | -             | -0.1377             |
| 3.0     | 231  | -             | 0.1226              |
| 4.0     | 308  | -             | 0.3958              |
| 5.0     | 385  | -             | 0.4814              |
| 6.0     | 462  | -             | 0.4691              |
| 6.4935  | 500  | 0.0243        | -                   |
| 7.0     | 539  | -             | 0.5681              |
| 8.0     | 616  | -             | 0.5352              |
| 9.0     | 693  | -             | 0.4855              |
| 10.0    | 770  | -             | 0.5279              |
| 11.0    | 847  | -             | 0.5321              |
| 12.0    | 924  | -             | 0.5120              |
| 12.9870 | 1000 | 0.0021        | -                   |
| 13.0    | 1001 | -             | 0.5190              |
| 14.0    | 1078 | -             | 0.5106              |
| 15.0    | 1155 | -             | 0.5265              |
| 16.0    | 1232 | -             | 0.5352              |
| 17.0    | 1309 | -             | 0.5098              |
| 18.0    | 1386 | -             | 0.5145              |
| 19.0    | 1463 | -             | 0.5786              |
| 19.4805 | 1500 | 0.0009        | -                   |
| 20.0    | 1540 | -             | 0.5759              |
| 21.0    | 1617 | -             | 0.4828              |
| 22.0    | 1694 | -             | 0.5396              |
| 23.0    | 1771 | -             | 0.5020              |
| 24.0    | 1848 | -             | 0.4970              |
| 25.0    | 1925 | -             | 0.5293              |
| 25.9740 | 2000 | 0.0008        | -                   |
| 26.0    | 2002 | -             | 0.5246              |
| 27.0    | 2079 | -             | 0.5460              |
| 28.0    | 2156 | -             | 0.5296              |
| 29.0    | 2233 | -             | 0.5201              |
| 30.0    | 2310 | -             | 0.5134              |
| 31.0    | 2387 | -             | 0.5435              |
| 32.0    | 2464 | -             | 0.5494              |
| 32.4675 | 2500 | 0.0006        | -                   |
| 33.0    | 2541 | -             | 0.5318              |
| 34.0    | 2618 | -             | 0.5405              |
| 35.0    | 2695 | -             | 0.5491              |
| 36.0    | 2772 | -             | 0.5522              |
| 37.0    | 2849 | -             | 0.5577              |
| 38.0    | 2926 | -             | 0.5572              |
| 38.9610 | 3000 | 0.0005        | -                   |
| 39.0    | 3003 | -             | 0.5552              |
| 40.0    | 3080 | -             | 0.5488              |
| 41.0    | 3157 | -             | 0.5552              |
| 42.0    | 3234 | -             | 0.5597              |
| 43.0    | 3311 | -             | 0.5569              |
| 44.0    | 3388 | -             | 0.5569              |
| 45.0    | 3465 | -             | 0.5547              |
| 45.4545 | 3500 | 0.0004        | -                   |
| 46.0    | 3542 | -             | 0.5513              |
| 47.0    | 3619 | -             | 0.5544              |
| 48.0    | 3696 | -             | 0.5602              |
| 49.0    | 3773 | -             | 0.5594              |
| 50.0    | 3850 | -             | 0.5594              |
| 1.0     | 128  | -             | 0.4619              |
| 2.0     | 256  | -             | 0.7729              |
| 3.0     | 384  | -             | 0.6288              |
| 3.9062  | 500  | 0.0157        | -                   |
| 4.0     | 512  | -             | 0.7186              |
| 5.0     | 640  | -             | 0.7690              |
| 6.0     | 768  | -             | 0.7236              |
| 7.0     | 896  | -             | 0.7844              |
| 7.8125  | 1000 | 0.0036        | -                   |
| 8.0     | 1024 | -             | 0.7651              |
| 9.0     | 1152 | -             | 0.7559              |
| 10.0    | 1280 | -             | 0.7707              |
| 11.0    | 1408 | -             | 0.8153              |
| 11.7188 | 1500 | 0.0022        | -                   |
| 12.0    | 1536 | -             | 0.8033              |
| 13.0    | 1664 | -             | 0.7598              |
| 14.0    | 1792 | -             | 0.7699              |
| 15.0    | 1920 | -             | 0.8298              |
| 15.625  | 2000 | 0.001         | -                   |
| 16.0    | 2048 | -             | 0.8317              |
| 17.0    | 2176 | -             | 0.8184              |
| 18.0    | 2304 | -             | 0.8206              |
| 19.0    | 2432 | -             | 0.8075              |
| 19.5312 | 2500 | 0.0007        | -                   |
| 20.0    | 2560 | -             | 0.8164              |
| 21.0    | 2688 | -             | 0.8289              |
| 22.0    | 2816 | -             | 0.8117              |
| 23.0    | 2944 | -             | 0.7980              |
| 23.4375 | 3000 | 0.0006        | -                   |
| 24.0    | 3072 | -             | 0.8326              |
| 25.0    | 3200 | -             | 0.8342              |
| 26.0    | 3328 | -             | 0.8083              |
| 27.0    | 3456 | -             | 0.8317              |
| 27.3438 | 3500 | 0.0005        | -                   |
| 28.0    | 3584 | -             | 0.8365              |
| 29.0    | 3712 | -             | 0.8334              |
| 30.0    | 3840 | -             | 0.8351              |
| 31.0    | 3968 | -             | 0.8055              |
| 31.25   | 4000 | 0.0005        | -                   |
| 32.0    | 4096 | -             | 0.8069              |
| 33.0    | 4224 | -             | 0.8289              |
| 34.0    | 4352 | -             | 0.8315              |
| 35.0    | 4480 | -             | 0.8412              |
| 35.1562 | 4500 | 0.0005        | -                   |
| 36.0    | 4608 | -             | 0.8418              |
| 37.0    | 4736 | -             | 0.8457              |

</details>

### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.5.2
- Datasets: 3.4.1
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### ContrastiveLoss
```bibtex
@inproceedings{hadsell2006dimensionality,
    author={Hadsell, R. and Chopra, S. and LeCun, Y.},
    booktitle={2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)},
    title={Dimensionality Reduction by Learning an Invariant Mapping},
    year={2006},
    volume={2},
    number={},
    pages={1735-1742},
    doi={10.1109/CVPR.2006.100}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->