# [What is Dataset Distillation Learning?](https://arxiv.org/abs/2406.04284)

After cloning the repo, download the distilled data and pretrained model from [Google Drive](https://drive.google.com/drive/folders/1kTQnt5WszAgifbyCYVPnaUYQrdqlAkNT?usp=sharing).
Afterwards, create the conda enviroment:
```
conda create -n learning python=3.10.12
conda activate learning
pip install -r requirements.txt
```
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
Use [jupyter notebook](https://github.com/princetonvisualai/What-is-Dataset-Distillation-Learning/blob/main/experiment_code/early_training_analysis/recognition.ipynb).

### Distilled data stores little information beyond what would be learned early in training
Refer to [jupyter notebook](https://github.com/princetonvisualai/What-is-Dataset-Distillation-Learning/blob/main/experiment_code/early_training_analysis/Hessian.ipynb). 

Note: computing Hessian approximation for the whole training data takes a long time (10+ hours on a L40 GPU). Comment out this Hessian computation if there is a lack of compute resources (Hessian calculations on distilled data is notably less resource intensive). 

## Semantics of Captured Information
Use [jupyter notebook](https://github.com/princetonvisualai/What-is-Dataset-Distillation-Learning/blob/main/experiment_code/influence_analysis/influence.ipynb) to generate the qualitative analysis (Figure 10) of the paper. For quantitative analysis, refer to the [LLaVa repo](https://github.com/haotian-liu/LLaVA). 
