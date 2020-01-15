# Adversarial Disentanglement with Grouped Observations

Source code for the experiments presented in the paper:

```
@inproceedings{nemeth2020,
  title = "Adversarial Disentanglement with Grouped Observations",
  author = "Jozsef Nemeth",
  booktitle={{Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20)}},
  year = "2020",
  arxivId = {2001.04761}
```

The repository is under construction, the current version is only for the MNIST experiments.

    
## Requirements
 * python3, tensorflow 1.13.1 or 1.14, matplotlib, sklearn
 
 
## Running Experiments
 * Extract mnist dataset.
 * Run `python3 test_mnist_trainings.py` to train models.
 * Run `python3 test_mnist_classify.py` for evaluation.
