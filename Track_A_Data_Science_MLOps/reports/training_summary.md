# Real Training Results

| model               |   accuracy |   precision |   recall |       f1 |   roc_auc | mlflow_run_id                    |
|:--------------------|-----------:|------------:|---------:|---------:|----------:|:---------------------------------|
| random_forest       |   0.754436 |    0.525641 | 0.76738  | 0.623913 |  0.841054 | 4f5533a22d6b4b98a65ccc233ca75a58 |
| logistic_regression |   0.738112 |    0.504303 | 0.783422 | 0.613613 |  0.841298 | 0259bc7823634527b06f5f1b55271ebb |
| gradient_boosting   |   0.805536 |    0.676056 | 0.513369 | 0.583587 |  0.845464 | ee37f6c228c747049e1da53476056a75 |

Selected **random_forest** by highest F1 (ROC-AUC tie-breaker).
