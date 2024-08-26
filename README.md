# MARVEL

The repo of the 39th IEEE/ACM International Conference on Automated Software Engineering (ASE'24) paper "Mutual Learning-Based Framework for Enhancing Robustness of Code Models via Adversarial Training", which contains the codes of our framework on all tasks and models.

---

## Overview

<img src="./marvel.png" alt="drawing" width="1000">

### Folder Structure

The folder structure is as follows.

```
.
├─language_parser
│  └─parser_folder
├─marvel
│  ├─authorshipAttribution
│  │  ├─code
│  │  ├─dataset
│  │  └─script
│  ├─defectPrediction
│  │  ├─code
│  │  ├─dataset
│  │  └─script
│  ├─Java250
│  │  ├─code
│  │  ├─dataset
│  │  └─script
│  ├─Python800
│  │  ├─code
│  │  ├─dataset
│  │  └─script
│  └─vulnerabilityDetection
│      ├─code
│      ├─dataset
│      └─script
└─utils
```

Under each subject's folder in `marvel/` (`authorshipAttribution/`, `defectPrediction/`, `Java250/`, `Python800/` and  `vulnerabilityDetection`), there are three folders (`code/`, `dataset/` and `script/`) and one `README.md` file. The original dataset and some data processing programs (for generating substitutions of attack algorithms) are stored in the `dataset/` directory. The `code/` directory contains our MARVEL codes, and attack codes (provided by [ALERT](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main) and [CODA](https://github.com/tianzhaotju/CODA/tree/main)), the `script/` directory contains commands for training and attacking.

### Dataset and Base Models

The experiments in our artifact use five open-source datasets, which have all been provided in their respective folders. The dataset files can be accessed from the following links: [Authorship Attribution](https://link.springer.com/chapter/10.1007/978-3-319-66402-6_6), [Defect Prediction](https://codechef.com), [Vulnerability Detection](https://proceedings.neurips.cc/paper_files/paper/2019/file/49265d2447bc3bbfe9e76306ce40a31f-Paper.pdf), [Java250 and Python800](https://github.com/IBM/Project_CodeNet).

In our artifact, we have applied MARVEL to three code models -  [CodeBERT](https://github.com/microsoft/CodeBERT/tree/master), [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT) and [UniXCoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) - to improve their robustness. We have also used three adversarial attack methods - [ALERT](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main), [MHM](https://github.com/Metropolis-Hastings-Modifier/MHM) and [CODA](https://github.com/tianzhaotju/CODA/tree/main) - to attack these models and evaluate their robustness.

## Setup

### Operating System and Hardware

The experiments related to CodeBERT and GraphCodeBERT were  conducted on an Ubuntu16.04 server with Intel(R) Xeon(R) E5 2683 v4 @2.10GHz CPU, and NVIDIA TITAN RTX GPU, and the experiments related to UniXCoder were conducted on an Ubuntu 20.04 server with Intel(R) Xeon(R) Platinum 8352V CPU@2.10GHz, and NVIDIA RTX 4090 GPU.

We recommend using a similar device, with at least one NVIDIA GPU on the device.

### Build `tree-sitter`

We use `tree-sitter` to parse code snippets and extract variable names. You need to go to `./language_parser/parser_folder` and build tree-sitter using the following commands:

```shell
bash build.sh
```



## Experiments

We applied MARVEL to three code models on five datasets and evaluated them using three attack algorithms. You can perform experiments on the corresponding downstream task in the `./marvel`.

We can refer to the ` README.md` files under each folder to train models on different tasks. Let's take the  `CodeBERT `model, `Authorship Attribution `task and the `ALERT `attack algorithm as examples. First, you need to go to `./marvel/authorshipAttribution`:

```
cd ./marvel/authorshipAttribution
```

Then make sure the dataset is already in the dataset folder, then go to `./code` , run python  `run_mutual.py` to execute the MARVEL framework as follows:

```shell
cd ./code
CUDA_VISIBLE_DEVICES=0 python run_mutual.py \
    --do_train \
    --do_test \
    --epochs=30 \
    --model_type codebert \
    --save_name marvel \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --alpha=0.3 \
    --max_adv_step=3
```

The trained model is then saved in the `./model/codebert` folder, then execute ALERT attack to attack the MARVEL-enhanced model:

```shell
CUDA_VISIBLE_DEVICES=0 python attack.py \
    --attack_name=alert \
    --model_type=codebert \
    --eval_batch_size=8 \
    --save_name=marvel
```

It should be noted that substitutions need to be generated based on the original paper setting of the corresponding attack algorithm before executing the corresponding attack algorithm (e.g., [ALERT](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main) and [CODA](https://github.com/tianzhaotju/CODA/tree/main)).

## Acknowledgement

We are very grateful that the authors of [CodeBERT](https://github.com/microsoft/CodeBERT/tree/master), [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT), [UniXCoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder), [ALERT](https://github.com/soarsmu/attack-pretrain-models-of-code/tree/main), [MHM](https://github.com/Metropolis-Hastings-Modifier/MHM), [CODA](https://github.com/tianzhaotju/CODA/tree/main) make their code publicly available so that we can build this repository on top of their code.
