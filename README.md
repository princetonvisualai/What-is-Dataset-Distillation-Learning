# [What is Dataset Distillation Learning?](https://arxiv.org/abs/2406.04284)

## Instructions
After cloning the repo, download the distilled data and pretrained model from [Google Drive](https://drive.google.com/drive/folders/1kTQnt5WszAgifbyCYVPnaUYQrdqlAkNT?usp=sharing).

## Distilled vs. Real Data
To run analyses done in Section 3 of the paper, refer to the two jupyter notebooks in [experiment_code/replacement_analysis](https://github.com/princetonvisualai/What-is-Dataset-Distillation-Learning/tree/main/experiment_code/replacement_analysis)

## Information Captured by Distilled Data
### Predictions of models trained on distilled data is similar to models trained with early-stopping
Generate a pool of subset-trained-models and early-stopped-models by running
```
python experiment_code/agreement_analysis/generate_models.py 
```
Finally, compare and visualize the prediction differences using the [jupyter notebook](https://github.com/princetonvisualai/What-is-Dataset-Distillation-Learning/blob/main/experiment_code/agreement_analysis/compare_models.ipynb).

### Recognition on the distilled data is learned early in the training process
Coming Soon.

### Distilled data stores little information beyond what would be learned early in training
Coming Soon.

## Semantics of Captured Information
Coming Soon.
