## Tennis ball tracking - non-regression

Contexte : `{"device": "cpu", "dataset_revision": "v1", "weights": "TENNIS_BALL_MODEL_04_11_2026.pt", "threshold": 0.5, "min_dist": 10}` - tolérance : 0.005

| Video                                                                                     | Metric    |   Current |   Baseline |   Delta | Status   |
|:------------------------------------------------------------------------------------------|:----------|----------:|-----------:|--------:|:---------|
| Fonseca vs Djokovic ｜ Full match ｜ Roland-Garros Round 3 2026 [n2zjuuvgkGY]_cut_0_825.mp4 | Accuracy  |    0.8715 |     0.8715 |  0.0000 | OK       |
| Fonseca vs Djokovic ｜ Full match ｜ Roland-Garros Round 3 2026 [n2zjuuvgkGY]_cut_0_825.mp4 | Precision |    0.9926 |     0.9926 |  0.0000 | OK       |
| Fonseca vs Djokovic ｜ Full match ｜ Roland-Garros Round 3 2026 [n2zjuuvgkGY]_cut_0_825.mp4 | Recall    |    0.7952 |     0.7952 |  0.0000 | OK       |
| Fonseca vs Djokovic ｜ Full match ｜ Roland-Garros Round 3 2026 [n2zjuuvgkGY]_cut_0_825.mp4 | F1-Score  |    0.8830 |     0.8830 |  0.0000 | OK       |
| TOTAL                                                                                     | Accuracy  |    0.8715 |     0.8715 |  0.0000 | OK       |
| TOTAL                                                                                     | Precision |    0.9926 |     0.9926 |  0.0000 | OK       |
| TOTAL                                                                                     | Recall    |    0.7952 |     0.7952 |  0.0000 | OK       |
| TOTAL                                                                                     | F1-Score  |    0.8830 |     0.8830 |  0.0000 | OK       |

### Comptages

| Video                                                                                     |   Total |   TP |   FP |   TN |   FN |
|:------------------------------------------------------------------------------------------|--------:|-----:|-----:|-----:|-----:|
| Fonseca vs Djokovic ｜ Full match ｜ Roland-Garros Round 3 2026 [n2zjuuvgkGY]_cut_0_825.mp4 |     825 |  400 |    3 |  319 |  103 |
| TOTAL                                                                                     |     825 |  400 |    3 |  319 |  103 |
