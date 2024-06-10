## Authorship Attribution Task

### Dataset

We have already provided the processed dataset in the `dataset` folder, you can skip this step.

### Fine-tune Original Code Models

To get to the original code model, i.e, without applying any robustness enhancement methods, run the following instructions:

```
cd script
bash train_original.sh
```

You can modify the `model_type` field to change the code model you want to train.

### Generate Substitution for the Attack Algorithms

For CODA attack, run the following instructions to get substitutions:

```
cd dataset
bash get_subs_coda.sh

```

For ALERT and MHM, you can use the file we provided, or run the following command:

```
cd dataset
bash get_subs_alert.sh
```

The codes for the attack algorithm comes from [ALERT](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main) and [CODA](https://github.com/tianzhaotju/CODA/tree/main), where you can refer to it for more details.

### Run MARVEL

To run marvel framework, run the following instructions:

```
cd script
bash train_marvel.sh
```

You can modify the `model_type` field to change the code model you want to train.

### Attack Original Model and MARVEL-Enhanced Model

```
cd script
bash attack_original.sh
bash attack_marvel.sh
```

You can modify the `model_type` and `attack_type` field to change the code model and attack algorithm you want to use.
