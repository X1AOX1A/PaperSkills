Title: MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning
ArXiv: 2406.09044
Authors: Hanqing Wang, , Yixia Li, , Shuo Wang, , Guanhua Chen, , Yun Chen, Shanghai University of Finance and Economics, Southern University of Science and Technology,, Tsinghua University, MoE Key Laboratory of Interdisciplinary Research of Computation and Economics,, Corresponding Authors.
Sections: 36
Estimated tokens: 19.4k

## Contents
- 1 Introduction
- 2 Preliminaries
  - Singular Value Decomposition
  - Low-Rank Adaptation
- 3 Methodology
- 4 Experiments
  - 4.1 Experiments on Large Language Model
    - Models and Datasets
    - Implementation Details
    - Results
  - 4.2 Experiments on Vision-Language Model
    - Models and Datasets
    - Implementation Details
    - Results
- 5 Understanding MiLoRA
  - 5.1 Comparison with More PEFTs
  - 5.2 Is Minor Singular Component Adaptation Important?
  - 5.3 How does the Matrix Update Δ ⁢ W Δ 𝑊 \Delta W roman_Δ italic_W Compare to W 𝑊 W italic_W ?
  - 5.4 MiLoRA Forgets Less than Baselines
  - 5.5 Comparison Between MiLoRA and PiSSA
- 6 Related Work
  - Parameter-Efficient Finetuning of LLMs.
  - LoRA and Its Variants
- 7 Conclusion
- Limitations
- Acknowledgements
- References
- Appendix A Detailed Experiment Setups
  - A.1 Similarity Metric Between Subspaces
  - A.2 Our Hyperparameter Setup for LLM
  - A.3 PiSSA Hyperparameter Setup for LLM
  - A.4 Hyperparameter Setup for Vision-Language Model
- Appendix B More Experiment Results
  - B.1 Experiments on Qwen2.5-7B
  - B.2 Complementary Weighet Metrics Analysis
- Appendix C Broader Impacts

## Abstract

Abstract Efficient finetuning of large language models (LLMs) aims to adapt the LLMs with reduced computational and memory costs. Previous LoRA-based approaches initialize the low-rank matrices with Gaussian distribution and zero values while keeping the original weight matrices frozen. However, the trainable model parameters optimized in an unguided subspace might interfere with the well-learned subspace of the pretrained weight matrices.
In this paper, we propose MiLoRA, a simple yet effective LLM finetuning approach that only updates the minor singular components of the weight matrix while keeping the principal singular components frozen. It is observed that the minor matrix corresponds to the noisy or long-tail information, while the principal matrix contains important knowledge. The MiLoRA initializes the low-rank matrices within a subspace that is orthogonal to the principal matrix, thus the pretrained knowledge is expected to be well preserved. During finetuning, MiLoRA makes the most use of the less-optimized subspace for learning the labeled dataset. Extensive experiments on commonsense reasoning, math reasoning, instruction following and visual instruction following benchmarks present the superior performance of our method.

## 1 Introduction

Large language models (Brown et al., 2020; Ouyang et al., 2022; Touvron et al., 2023; Jiang et al., 2023, LLMs) have demonstrated superior performance on various tasks (Zheng et al., 2023), such as math reasoning (Wang et al., 2024b) and question answering (Ivison et al., 2023). These models are pretrained with the next token prediction task (Brown et al., 2020) on large web-scale data, then finetuned with instruction data as well as human preference data (Ouyang et al., 2022; Yu et al., 2023; Cui et al., 2023) for different downstream tasks.
Fully finetuning is commonly employed to unlock the complete potential of LLMs, however, optimizing all model parameters necessitates substantial and restrictive computational resources (Touvron et al., 2023; Jiang et al., 2023), which hampers the utilization of LLMs across diverse scenarios.

Parameter-efficient finetuning (Hu et al., 2021, 2023; Liu et al., 2022; Zhang et al., 2024; Liu et al., 2024b, PEFT) aims at reducing the computational and GPU memory cost for finetuning of pretrained models. Low-rank adaptation (Hu et al., 2021, LoRA) is one of the most widely used PEFT methods for LLM finetuning. It assumes the change of linear model weights to be low-rank (Li et al., 2018; Aghajanyan et al., 2021). For each selected weight matrix, it only updates two low-rank matrices while keeping the pretrained weight frozen. During inference, the low-rank matrices are merged into the pretrained linear weights, thus no additional computational or memory cost is introduced.
Recently, researchers have explored different LoRA-based variants for efficient LLM finetuning (Zhang et al., 2023; Pan et al., 2024; Kopiczko et al., 2024; Liu et al., 2024b; Meng et al., 2024). However, most existing LoRA-based methods randomly initialize the low-rank matrices and optimize the trainable parameters in an unguided subspace. We suspect this strategy may override important pretrained features, thus degrading the performance of low-rank adaptation methods (Dou et al., 2024).

In this paper, we propose Minor singular component based Low Rank Adaptation (MiLoRA) for efficient LLM finetuning. MiLoRA has a similar framework as LoRA but employs a different initialization schedule. Specifically, a weight matrix ${\bm{W}}$ is decomposed with the singular value decomposition (SVD) algorithm. Based on the magnitude of the singular values, we divide ${\bm{W}}$ into two components: the principal matrix ${\bm{W}}_{p}$ corresponding to large singular values and the minor matrix ${\bm{W}}_{m}$ corresponding to small singular values.
We argue that the principal matrix captures the essence of the pretrained knowledge, whereas the minor matrix is suboptimal with noisy or long-tail information.
It is supported by previous works (Hajimolahoseini et al., 2021; Sharma et al., 2024; Li et al., 2024) that principal low-rank approximation can achieve comparable or even better performance to full parameters.

Motivated by these observations, we freeze the principal matrix ${\bm{W}}_{p}$ and adapt the minor singular components during finetuning. The low-rank matrices ${\bm{A}}$ and ${\bm{B}}$ in LoRA framework are initialized with the minor matrix ${\bm{W}}_{m}$.
Since the trainable low-rank matrices are initialized in a subspace orthogonal to the principal matrix, MiLoRA is expected to effectively learn from finetuning tasks while better preserving and utilizing the pretrained knowledge.
To maintain the capability of the pretrained model at the start of finetuning, vanilla LoRA explicitly initializes ${\bm{B}}$ with zeros. In contrast, MiLoRA naturally satisfies this requirement, as the pretrained weight matrix ${\bm{W}}$ equals the frozen principal part ${\bm{W}}_{p}$ plus the low-rank part ${\bm{W}}_{m}={\bm{B}}{\bm{A}}$.
We conduct extensive experiments on commonsense reasoning, math reasoning, instruction following and visual instruction following benchmarks. The experimental results show that MiLoRA consistently outperforms LoRA without sacrificing training or inference efficiency, such
as commonsense reasoning (+1.6/+1.1 on LLaMA2-7B/LLaMA3-8B), math reasoning (+2.0 on LLaMA2-7B), instruction following (+2.9 on LLaMA2-7B) and visual instruction following (+1.4 on LLaVA1.5-7B).(^1^11Our code and model are publicly available at [https://github.com/sufenlp/MiLoRA](https://github.com/sufenlp/MiLoRA).)

## 2 Preliminaries

#### Singular Value Decomposition

Given a matrix ${\bm{W}}\in\mathbb{R}^{m\times n}$, its singular value decomposition is denoted as ${\bm{W}}={\bm{U}}\bm{\Sigma}{\bm{V}}^{\top}$, where ${\bm{U}}=\left[u_{1},u_{2},\cdots,u_{m}\right]\in\mathbb{R}^{m\times m}$, ${\bm{V}}=\left[v_{1},v_{2},\cdots,v_{n}\right]\in\mathbb{R}^{n\times n}$. The columns of ${\bm{U}}$ are the left singular vectors, and the columns of ${\bm{V}}$ are the right singular vectors. The $\bm{\Sigma}\in\mathbb{R}^{m\times n}$ is a diagonal matrix containing the singular values of ${\bm{W}}$ in descending order.
Without loss of generality, we suppose $m\leq n$ to simplify the notation. The SVD of ${\bm{W}}$ can be reformulated as

$$ $\displaystyle{\bm{W}}={\bm{U}}\bm{\Sigma}{\bm{V}}^{\top}=\sum_{i=1}^{m}\sigma_ {i}u_{i}v_{i}^{\top},$ (1) $$

where $u_{i}$ and $v_{i}$ are the $i^{\mathrm{t}h}$ column of ${\bm{U}}$ and ${\bm{V}}$, respectively.

#### Low-Rank Adaptation

The low-rank adaptation method (Hu et al., 2021, LoRA) assumes the updates of linear weight ${\bm{W}}\in\mathbb{R}^{m\times n}$ to be low-rank, thus models the changes with two trainable low-rank matrices ${\bm{A}}\in\mathbb{R}^{r\times n}$ and ${\bm{B}}\in\mathbb{R}^{m\times r}$. The weight matrix can be decomposed as

$$ $\displaystyle{\bm{W}}={\bm{W}}^{(0)}+\Delta{\bm{W}}={\bm{W}}^{(0)}+\frac{ \alpha}{r}{\bm{B}}{\bm{A}},$ (2) $$

where ${\bm{W}}^{(0)}$ and $\Delta{\bm{W}}$ refer to the pretrained weight and weight change, respectively. The $\alpha$ and $r$ are hyperparameters of scaling-factor and LoRA rank ($r\ll\min(m,n)$).
During finetuning, the pretrained matrix ${\bm{W}}^{(0)}$ is kept frozen. It significantly diminishes the number of trainable parameters as both ${\bm{A}}$ and ${\bm{B}}$ matrices are low-rank.
The ${\bm{B}}$ matrix is initialized with zero while ${\bm{A}}$ matrix adopts a random Gaussian distribution with zero mean value.
This initialization strategy ensures the $\Delta{\bm{W}}=0$ at the beginning of training. The LoRA method only modifies the linear matrices in the Transformer model. The low-rank matrices can be easily merged into the pretrained linear matrix to get updated for inference, which does not require additional computing and GPU memory compared with full finetuning.
However, the vanilla LoRA method fails to select the optimal subspace for updating the model parameters, as the low-rank matrices ${\bm{A}}$ and ${\bm{B}}$ are randomly initialized. This might potentially detract from the pretrained knowledge encoded in the pretrained weight matrix.

Figure: Figure 1: (a) MiLoRA method splits the pretrained linear weight matrix into two parts, the minor singular component (${\bm{U}}_{m},{\bm{V}}_{m},{\bm{\Sigma}}_{m}$) is used to initialize the low-rank matrices ${\bm{A}}_{m}$ and ${\bm{B}}_{m}$ in the LoRA framework. (b) During finetuning, only the low-rank matrices ${\bm{A}}_{m}$ and ${\bm{B}}_{m}$ are updated while the principal matrix ${\bm{W}}_{p}$ is frozen, which is similar to the practice of LoRA method.
Refer to caption: x1.png

## 3 Methodology

The proposed MiLoRA is a simple yet effective PEFT approach that selects the minor singular components of the weight matrices for optimization. As observed in LASER (Sharma et al., 2024), the minor singular components of weight matrices contain noisy or long-tail information, while the principal singular components are responsible for important features across tasks. Therefore, MiLoRA is designed to effectively learn from the finetuning dataset by adapting minor singular components, while maintaining the knowledge encoded in the pretrained model. On the contrary, vanilla LoRA fails to constrain the optimization subspace as its low-rank matrices are randomly initialized.
Specifically, suppose $m\leq n$, MiLoRA splits each linear weight matrix into two matrices according to the corresponding singular values: principal matrix ${\bm{W}}_{p}$ and minor matrix ${\bm{W}}_{m}$ (see Figure [1](https://arxiv.org/html/2406.09044v3#S2.F1)):

$$ $\displaystyle{\bm{W}}$ $\displaystyle={\bm{W}}_{p}+{\bm{W}}_{m}$ (3) $\displaystyle=\sum_{i=1}^{m-r}\sigma_{i}u_{i}v_{i}^{\top}+\sum_{i=m-r+1}^{m} \sigma_{i}u_{i}v_{i}^{\top},$ $$

where the singular values are in descending order ($\sigma_{1}\geq\sigma_{2}\geq\cdots\geq\sigma_{m}\geq 0$), and the hyperparameter $r$ is the number of minor singular values in the ${\bm{W}}_{m}$ matrix.

The decomposition can also be reformulated in a matrix form. The ${\bm{U}}$ matrix in the SVD of ${\bm{W}}$ can be reformulated as ${\bm{U}}=\left[{\bm{U}}_{p},{\bm{U}}_{m}\right]$, where ${\bm{U}}_{p}=\left[u_{1},u_{2},\cdots,u_{m-r}\right]$ and ${\bm{U}}_{m}=\left[u_{m-r+1},u_{m-r+2},\cdots,u_{m}\right]$ are left singular vectors corresponding to principal and minor singular values, respectively. The ${\bm{V}}$ and ${\bm{\Sigma}}$ matrices can be reformulated similarly. The SVD of ${\bm{W}}$ can be expressed as:

$$ $\displaystyle{\bm{W}}$ $\displaystyle={\bm{U}}{\bm{\Sigma}}{\bm{V}}^{\top}={\bm{U}}_{p}{\bm{\Sigma}}_{ p}{\bm{V}}_{p}^{\top}+{\bm{U}}_{m}{\bm{\Sigma}}_{m}{\bm{V}}_{m}^{\top}$ (4) $\displaystyle={\bm{W}}_{p}+{\bm{W}}_{m}$ $$

During the finetuning process, instead of freezing the entire pretrained weight matrix, we just keep the principal singular components ${\bm{W}}_{p}$ fixed to preserve the pretrained knowledge. The minor matrix ${\bm{W}}_{m}$ is used to initialize low-rank matrices ${\bm{A}}_{m}$ and ${\bm{B}}_{m}$ (see Figure [1](https://arxiv.org/html/2406.09044v3#S2.F1)):

$$ $\displaystyle{\bm{W}}_{m}$ $\displaystyle={\bm{U}}_{m}{\bm{\Sigma}}_{m}{\bm{V}}_{m}^{\top}$ (5) $\displaystyle=({\bm{U}}_{m}\sqrt{{\bm{\Sigma}}_{m}})(\sqrt{{\bm{\Sigma}}_{m}}{ \bm{V}}_{m}^{\top})={\bm{B}}_{m}{\bm{A}}_{m},$ $$

This strategy has two benefits: 1) It encourages the model to learn in the less-optimized subspace spanned by the minor singular vectors, thus reducing the interference with the well-learned pretrained knowledge encoded in the principal singular components. 2) Unlike vanilla LoRA which requires tuning the scaling factor $\alpha$ and the initialization hyperparameters, ours does not require any hyperparameter except the rank $r$.

One potential variant of MiLoRA is to impose strict constraints on the orthogonality between ${\bm{W}}_{m}$ and ${\bm{W}}_{p}$ during fine-tuning. However, we adopt a softer approach by guiding the optimization direction solely through initialization, resulting in a method that is simpler, more training-efficient, and more flexible. Our experiment results show that our scheme works quite well in balancing learning from the finetuning dataset and preserving pretrained knowledge.

## 4 Experiments

**Table 1: Commonsense reasoning evaluation results for LLaMA2-7B and LLaMA3-8B on eight tasks. The reported metric in this table is accuracy. ^†Results are cited from Liu et al. (2024b). Bold numbers indicate the highest performance scores for each dataset across the different PEFT methods for the corresponding model.**
| Model | PEFT | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ChatGPT | $-$ | 73.1 | 85.4 | 68.5 | 78.5 | 66.1 | 89.8 | 79.9 | 74.8 | 77.0 |
| LLaMA2-7B | LoRA^† | 69.8 | 79.9 | 79.5 | 83.6 | 82.6 | 79.8 | 64.7 | 81.0 | 77.6 |
| PiSSA | 67.6 | 78.1 | 78.4 | 76.6 | 78.0 | 75.8 | 60.2 | 75.6 | 73.8 |  |
| MiLoRA | 67.6 | 83.8 | 80.1 | 88.2 | 82.0 | 82.8 | 68.8 | 80.6 | 79.2 |  |
| LLaMA3-8B | LoRA^† | 70.8 | 85.2 | 79.9 | 91.7 | 84.3 | 84.2 | 71.2 | 79.0 | 80.8 |
| PiSSA | 67.1 | 81.1 | 77.2 | 83.6 | 78.9 | 77.7 | 63.2 | 74.6 | 75.4 |  |
| MiLoRA | 68.8 | 86.7 | 77.2 | 92.9 | 85.6 | 86.8 | 75.5 | 81.8 | 81.9 |  |

**Table 2: Math reasoning evaluation results for LLaMA2-7B on GSM8K and MATH. The evaluation metric presented in this table is the Exact Match (EM) ratio. ^†Results are cited from Yu et al. (2023). For each dataset, the highest performance scores among the different PEFT methods are shown in bold.**
| Method | GSM8K | MATH | Avg. |
| --- | --- | --- | --- |
| Full FT^† | 66.5 | 19.8 | 43.2 |
| LoRA | 60.6 | 16.9 | 38.7 |
| PiSSA | 58.2 | 15.8 | 37.0 |
| MiLoRA | 63.5 | 17.8 | 40.7 |

To evaluate MiLoRA against other PEFTs, we conduct experiments on both NLP and multimodal tasks covering 18 datasets. We benchmark against LoRA (Hu et al., 2021) and PiSSA (Meng et al., 2024). All of our experiments are conducted on 8 NVIDIA L40 GPUs unless otherwise specified.

- •
LoRA (Hu et al., 2021) reparameterizes the weight update $\Delta{\bm{W}}$ with two trainable low-rank matrices ${\bm{A}}$ and ${\bm{B}}$, while freezing the pretrained weight ${\bm{W}}$. They use a random Gaussian initialization for A and zero for B, so $\Delta{\bm{W}}={\bm{B}}{\bm{A}}$ is zero at the beginning of training.
- •
PiSSA (Meng et al., 2024) shares the same framework as LoRA, but employs a more sophisticated initialization approach. They initialize ${\bm{A}}$ and ${\bm{B}}$ with principal singular values and singular vectors of the pre-trained weight ${\bm{W}}$.
Given that the principal components capture the essence of a matrix, PiSSA is expected to better approximate full finetuning by changing the essential parts while freezing the “noisy” parts.

### 4.1 Experiments on Large Language Model

#### Models and Datasets

We compare MiLoRA with baselines on three different types of LLM downstream tasks.

$\bullet\,$Commonsense reasoning: We finetune LLaMA2-7B (Touvron et al., 2023) and LLaMA3-7B (AI@Meta, 2024) on Commonsense170K (Hu et al., 2023).
Eight commonsense reasoning datasets are used for evaluation, including BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021), ARC-e, ARC-c (Clark et al., 2018), and OBQA (Mihaylov et al., 2018). The task is formulated as a multiple-choice problem. We report accuracy (%) for all datasets on the best checkpoint chosen by the validation set loss.

$\bullet\,$Math reasoning: We finetune LLaMA2-7B (Touvron et al., 2023) on the MetaMathQA dataset (Yu et al., 2023), which contains 395K samples augmented from the training set of GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021).
We use test sets of GSM8K and MATH for evaluation and report results on the last checkpoint.
We report the Exact Match (EM) ratio against the ground truth for each test set.

$\bullet\,$Instruction following: We finetune LLaMA2-7B with Ultrafeedback  (Cui et al., 2023) following previous works Wu et al. (2024a, b). We evaluate our models using AlpacaEval 2.0 (Dubois et al., 2024; Li et al., 2023a), FollowBench (Jiang et al., 2024), and IFEval (Zhou et al., 2023). For AlpacaEval 2.0, we use Length-controlled Win Rate (LC WR) and Win Rate (WR) against GPT-4 Turbo (OpenAI, 2023b) as evaluation metrics. In FollowBench, we adopt Hard Satisfaction Rate (HSR), while in IFEval, we measure accuracy.

**Table 3: Instruction following evaluation results for LLaMA2-7B on AlpacaEval 2.0, FollowBench, and IFEval. AlpacaEval 2.0 is evaluated using Length-controlled Win Rate (LC WR) and Win Rate (WR) against GPT-4 Turbo (OpenAI, 2023b), FollowBench adopts Hard Satisfaction Rate (HSR), and IFEval is measured by accuracy. Bold numbers indicate the highest performance scores for each dataset across the different PEFT methods for the corresponding model.**
| Method | AlpacaEval 2.0 | FollowBench | IFEval | Avg. |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LC WR | WR | Level1 | Level2 | Level3 | Level4 | Level5 | Prompt-strict | Ins-strict | Prompt-loose | Ins-loose |  |  |
| LoRA | 5.6 | 3.9 | 46.5 | 39.2 | 35.5 | 22.2 | 14.3 | 28.8 | 39.9 | 31.1 | 42.3 | 28.1 |
| PiSSA | 5.6 | 3.9 | 44.2 | 37.1 | 30.6 | 20.9 | 14.3 | 31.6 | 42.7 | 34.6 | 45.7 | 28.3 |
| MiLoRA | 7.1 | 4.4 | 45.1 | 37.6 | 38.2 | 28.5 | 20.8 | 32.4 | 44.8 | 34.8 | 47.7 | 31.0 |

#### Implementation Details

We use the same hyperparameter configurations as Hu et al. (2023) without tuning for all methods. Details can be found in Appendix [A.2](https://arxiv.org/html/2406.09044v3#A1.SS2). We denote this hyperparameter setup as LLM-Adapters. By default, we use a rank of 32. As math reasoning has a large training dataset, we set the rank to 64. We use the implementation of LLM-Adapters (Hu et al., 2023)(^2^22[https://github.com/AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)) for commonsense reasoning, the implementation of PiSSA (Meng et al., 2024)(^3^33[https://github.com/GraphPKU/PiSSA](https://github.com/GraphPKU/PiSSA)) for math reasoning and the implementation of open-instruct(^4^44[https://github.com/allenai/open-instruct](https://github.com/allenai/open-instruct)) for instruction tuning.

#### Results

**Table 4: Visual instruction tuning evaluation results for LLaVA1.5-7B on seven vision-language tasks. Accuracy serves as the metric reported in this table. ^†We directly use checkpoints from Liu et al. (2024a) to reproduce the results of baseline methods.**
| Method | VQAv2 | GQA | VisWiz | SQA | VQAT | POPE | MMBench | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full FT^† | 78.5 | 61.9 | 50.0 | 66.8 | 58.2 | 85.9 | 64.3 | 66.5 |
| LoRA^† | 79.1 | 62.9 | 47.8 | 68.4 | 58.2 | 86.4 | 66.1 | 66.9 |
| PiSSA | 77.5 | 60.6 | 39.2 | 67.3 | 54.3 | 87.0 | 61.4 | 63.9 |
| MiLoRA | 79.2 | 62.1 | 53.3 | 70.6 | 58.7 | 87.9 | 66.1 | 68.3 |

We report results in Table [1](https://arxiv.org/html/2406.09044v3#S4.T1) for commonsense reasoning, Table [2](https://arxiv.org/html/2406.09044v3#S4.T2) for math reasoning, and Tables [3](https://arxiv.org/html/2406.09044v3#S4.T3) for instruction following, respectively. For commonsense reasoning, we also include the ChatGPT baseline reported in Liu et al. (2024b) as a reference, which is obtained with GPT-3.5 Turbo (OpenAI, 2023a) API using a zero-shot Chain of Thought (Wei et al., 2022). For math reasoning, we add the full finetuning results from Yu et al. (2023) as a reference.

As can be seen, MiLoRA consistently surpasses all baseline methods across different datasets and LLMs, indicating that MiLoRA is a highly effective finetuning method. Specifically, for commonsense reasoning, MiLoRA outperforms LoRA and PiSSA by an average of 1.6 and 5.4 accuracy points on LLaMA2-7B and 1.1 and 6.5 points on LLaMA3-8B. For math reasoning, MiLoRA improves over LoRA and PiSSA by 2.0 and 3.7 averaged Exact Match (EM) points on LLaMA2-7B, respectively. However, it still underperforms full finetuning by an average of 2.5 EM score. This suggests that the PEFT methods still have room for improvement to fully match the performance of full fine-tuning. For instruction following, the results on AlpacaEval 2.0 indicate that PiSSA performs comparably to LoRA, while MiLoRA outperforms both by 1.5 in LC WR and 0.5 in WR, demonstrating its superior effectiveness. Furthermore, experiments on FollowBench and IFEval show that MiLoRA significantly surpasses both LoRA and PiSSA across these benchmarks. Notably, MiLoRA excels on more challenging problems (levels 4 and 5) in FollowBench, likely due to its ability to better retain pre-trained knowledge, which is crucial for handling complex instructions. Overall, MiLoRA surpasses LoRA and PiSSA in instruction following, achieving an average improvement of 2.9 and 2.7 points, respectively, further underscoring its advantages.

### 4.2 Experiments on Vision-Language Model

#### Models and Datasets

To further examine if MiLoRA can remain competitive on multimodal finetuning tasks, we further conduct visual instruction tuning tasks on LLaVA1.5-7B Liu et al. (2024a), which consists of an LLM, Vicuna-1.5-7B Peng et al. (2023) and a vision encoder, CLIP ViT-L/336px Radford et al. (2021). The visual instruction tuning is performed in a multitask setting, including VQA Hudson and Manning (2019); Marino et al. (2019); Schwenk et al. (2022), OCR Mishra et al. (2019); Sidorov et al. (2020), region-level VQA Krishna et al. (2017); Mao et al. (2016), visual conversation Liu et al. (2024a), and language conversation data. The finetuned models are evaluated on seven different multimodal benchmarks: VQA-v2 Goyal et al. (2017), GQA Hudson and Manning (2019), VizWiz Gurari et al. (2018), SQA Lu et al. (2024), VQAT Singh et al. (2019), POPE Li et al. (2023b), and MMBench Liu et al. (2024c).

#### Implementation Details

We follow the settings of Liu et al. (2024a) to construct the training data and prompt template. For the hyperparameter configuration, we follow the LoRA configuration provided by Liu et al. (2024a) without tuning for all PEFT methods for a fair comparison. Details can be found in Appendix [A.4](https://arxiv.org/html/2406.09044v3#A1.SS4).

#### Results

Table [4](https://arxiv.org/html/2406.09044v3#S4.T4) shows the experiment results. As can be seen, MiLoRA achieves the best results, outperforming LoRA and PiSSA by 1.4 and 4.4 average accuracy points, respectively. We also note that in this setup, fully fine-tuning is less effective than LoRA. As a result, methods like PiSSA, which aim to closely approximate fully fine-tuning, may lose their advantage. In contrast, MiLoRA improves performance by retaining more of the pre-trained knowledge, thereby circumventing this issue.

## 5 Understanding MiLoRA

In this section, we conduct experiments to further understand MiLoRA. By default, we use the finetuned LLaMA2-7B model on math reasoning from our main experiments for analyses.

### 5.1 Comparison with More PEFTs

**Table 5: Comparison with more baselines on math and code tasks. Baseline GSM8K and Human-eval results are cited from Wang et al. (2024c). The evaluation metric for GSM8K is Exact Match (EM), while for Human-eval, it is Pass@1.**
| Method | GSM8K | Human-eval | Training Cost |
| --- | --- | --- | --- |
| rsLoRA | 45.6 | 16.0 | $\times$ 1.00 |
| LoRA+ | 52.1 | 18.2 | $\times$ 1.00 |
| DoRA | 53.1 | 19.8 | $\times$ 1.58 |
| AdaLoRA | 50.7 | 17.8 | $\times$ 1.07 |
| LoRA-GA | 53.6 | 19.8 | $\times$ 1.00 |
| MiLoRA | 54.7 | 24.4 | $\times$ 1.00 |

To further demonstrate the effectiveness of MiLoRA, we compare MiLoRA with additional popular baselines, such as rsLoRA Kalajdzievski (2023), LoRA+ Hayou et al. (2024), DoRA Liu et al. (2024b), AdaLoRA Zhang et al. (2023), LoRA-GA Wang et al. (2024c). We follow the math and code experiment settings in Wang et al. (2024c). Specifically, we finetune LLaMA2-7B with MiLoRA on MetaMathQA 100K and Code-Feedback 100K for one epoch using rank $8$ and compare MiLoRA with baseline results reported in Wang et al. (2024c). Table [5](https://arxiv.org/html/2406.09044v3#S5.T5) shows the results. We report the training cost of all methods relative to LoRA, which requires 3.58 hours of training on the math task using a single NVIDIA-L40 GPU. As can be seen, MiLoRA consistently outperforms all baselines, improving the best PEFT baseline LoRA-GA by 1.1 and 4.6 points on GSM8K and Human-eval, respectively. The training efficiency of MiLoRA is also very high. The time for SVD decomposition in MiLoRA is less than six minutes and negligible when compared with the total training time.
Therefore, the training time required of MiLoRA is consistent with that of LoRA, while DoRA and AdaLoRA require additional training time overhead.

### 5.2 Is Minor Singular Component Adaptation Important?

**Table 6: Performance of MiLoRA when initializing with principal, random sampled, and minor singular components.**
| Method | Principal | Random | Minor |
| --- | --- | --- | --- |
| GSM8K | 60.7 | 63.2 | 64.0 |
| MATH | 14.6 | 15.5 | 16.1 |

To investigate the influence of singular components of varying magnitudes on finetuning performance, we conduct experiments on math reasoning tasks using the LLaMA2-7B model. Specifically, we finetune LLaMA2-7B on the MetaMathQA 395K dataset for 1 epoch with a maximum sequence length of 512 to save computation. We initialize the low-rank matrices ${\bm{A}}$ and ${\bm{B}}$ with principal, minor, and randomly sampled singular components, and report the evaluation results in Table [6](https://arxiv.org/html/2406.09044v3#S5.T6). As can be seen, initializing with the minor singular components achieves the best performance across both datasets.
This underscores the importance of adapting the minor singular components during the finetuning process.

**Table 7: The Frobenius norm of ${\bm{U}}^{\top}{\bm{W}}{\bm{V}}$ where ${\bm{U}}$ and ${\bm{V}}$ are the left/right top $r$ singular vectors of either ${\bm{W}}$, a random matrix, or $\Delta{\bm{W}}$ of PEFTs. The weight matrices are taken from the query projection of the 16th layer of LLaMA2-7B.**
| $\left\|{\bm{W}}\right\|_{F}=97.36$ | ${\bm{W}}$ | Random | LoRA | PiSSA | MiLoRA |
| --- | --- | --- | --- | --- | --- |
| $\left\|{\bm{U}}^{\top}{\bm{W}}{\bm{V}}\right\|_{F}=$ | 43.64 | 1.50 | 1.82 | 17.57 | 1.86 |
| $\left\|{\bm{U}}^{\top}\Delta{\bm{W}}{\bm{V}}\right\|_{F}=$ | - | - | 68.18 | 55.79 | 44.95 |
| Amplification factor | - | - | 37.46 | 3.18 | 24.17 |

Figure: Figure 2: Left: The subspace similarity averaged by all modules in a layer. Right: The subspace similarity averaged by all layers for a specific module type.
Refer to caption: x2.png

### 5.3 How does the Matrix Update Δ ⁢ W Δ 𝑊 \Delta W roman_Δ italic_W Compare to W 𝑊 W italic_W ?

We investigate the relationship between $\Delta{\bm{W}}$ and ${\bm{W}}$ of different methods, where $\Delta{\bm{W}}$ is the difference between the finetuned and pre-trained weights. Following the analysis method in Hu et al. (2021), we project ${\bm{W}}$ and $\Delta{\bm{W}}$ onto the $r$-dimensional subspace of $\Delta{\bm{W}}$ by computing ${\bm{U}}^{T}{\bm{W}}{\bm{V}}$ and ${\bm{U}}^{T}\Delta{\bm{W}}{\bm{V}}$, with ${\bm{U}}$/${\bm{V}}$ being the left/right top $r$ singular-vector matrix of $\Delta{\bm{W}}$. As a comparison, we also compute ${\bm{U}}^{T}{\bm{W}}{\bm{V}}$ by replacing ${\bm{U}}$, ${\bm{V}}$ with the top $r$ singular vectors of ${\bm{W}}$ or a random matrix. The results are shown in Table [7](https://arxiv.org/html/2406.09044v3#S5.T7). Following Hu et al. (2021), we use the query weight in the middle layer (16th) of the model for analysis and use the Frobenius norm as the measurement of magnitude.

We make two key observations. First, $\Delta{\bm{W}}$ of LoRA and MiLoRA have a similar correlation with ${\bm{W}}$, both slightly stronger than a random matrix, indicating that they amplify directions not emphasized in ${\bm{W}}$. In contrast, $\Delta{\bm{W}}$ of PiSSA shows a strong correlation with ${\bm{W}}$. This is attributed to PiSSA’s direct optimization of the principal singular components. Second, the amplification factor varies significantly across different methods. Compared to LoRA, we find the amplification factor of MiLoRA is significantly smaller. This indicates that MiLoRA has a reduced impact on the features which already present in ${\bm{W}}$. See Table [14](https://arxiv.org/html/2406.09044v3#A2.T14) of Appendix [B.2](https://arxiv.org/html/2406.09044v3#A2.SS2) for analyzing the weight matrix of MLP down projection in the same layer, which demonstrates similar trends.

To further analyze the characteristics of $\Delta{\bm{W}}$ for different methods, we measure the normalized subspace similarity (Hu et al., 2021) between the subspaces spanned by top $r$ singular vectors of $\Delta{\bm{W}}$ and the subspace spanned by top $r$ singular vectors, bottom $r$ singular vectors, and a random $r$ singular vectors of ${\bm{W}}$. We define the normalized similarity metric based on the Grassmann distance following Hu et al. (2021) (See Appendix [A.1](https://arxiv.org/html/2406.09044v3#A1.SS1) for more details) and report the results in Figure [2](https://arxiv.org/html/2406.09044v3#S5.F2). From the left subfigure, it is clear that PiSSA learns in directions closely aligned with the top singular vectors of the pretrained weights, while LoRA and MiLoRA do not significantly optimize in these directions. Specifically, MiLoRA focuses on optimizing directions associated with the bottom singular vectors of the pretrained weights. The right subfigure further shows that weight matrices in both MLP and self attention modules of MiLoRA learn in directions more closely aligned with the bottom singular vectors, especially the MLP modules.

### 5.4 MiLoRA Forgets Less than Baselines

**Table 8: Forgetting loss of various PEFT methods.**
| Method | LoRA | PiSSA | MiLoRA |
| --- | --- | --- | --- |
| Forgetting loss | 3.24 | 6.07 | 2.54 |

A hypothesis explanation for why MiLoRA outperforms baselines is that pre-trained knowledge is more retained. To test this hypothesis, we follow Kalajdzievski (2024) and use cross-entropy as the metric for measuring forgetting. This is the usual next token prediction loss used when training LLMs, except that the target next token is replaced by the distribution predicted by the pre-trained base model. We evaluate the forgetting metric on the LLaMA2-7B model finetuned in math reasoning using the WikiText-103 test dataset following Kalajdzievski (2024). As shown in Table [8](https://arxiv.org/html/2406.09044v3#S5.T8), MiLoRA exhibits the lowest forgetting loss, which is consistent with our hypothesis that MiLoRA makes the least modification to pre-trained knowledge. In contrast, the loss of PiSSA is significantly higher than that of the other methods.

### 5.5 Comparison Between MiLoRA and PiSSA

**Table 9: Math reasoning evaluation results for LLaMA2-7B with PiSSA hyperparameters and our (LLM-Adapters) hyperparameters.**
| Hyp. Setup | PEFT | GSM8K | MATH | Avg. |
| --- | --- | --- | --- | --- |
| PiSSA | LoRA | 41.5 | 5.8 | 23.6 |
| PiSSA | 51.3 | 7.6 | 29.4 |  |
| MiLoRA | 40.0 | 5.2 | 22.6 |  |
| LLM-Adapters | LoRA | 56.6 | 10.8 | 33.7 |
| PiSSA | 51.3 | 10.4 | 30.8 |  |
| MiLoRA | 58.6 | 11.6 | 35.1 |  |

Concurrent with our research, Meng et al. (2024) have recently introduced a low-rank adaptation method called PiSSA. PiSSA shares a similar framework as MiLoRA, but adapts the principal singular components. We argue that PiSSA and MiLoRA are fundamentally different.

Motivation: PiSSA is designed to approximate full finetuning by adapting the principal singular components, which are believed to capture the essence of the weight matrices. In contrast, our method MiLoRA aims to adapt to new tasks while maximally retaining the base model’s knowledge. To achieve this, we instead finetune the minor singular components of the weight matrices, which are less important for the pretrained weight matrices.

Performance: The PiSSA paper claims that PiSSA outperforms both LoRA and adapting minor singular components. However, our investigation suggests that this claim is likely due to the specific hyperparameters used in their experiments. To verify this, we replicate their experimental setup using the same MetaMathQA 100K dataset and compare the performance of PiSSA, LoRA, and MiLoRA under both PiSSA’s hyperparameters and our own. Specifically, we choose a rank of 128 where PiSSA obtains the best performance in their paper and finetunes for 1 epoch for all setups following PiSSA paper. The results are summarized in Table [9](https://arxiv.org/html/2406.09044v3#S5.T9).

We have several observations. First, under the PiSSA hyperparameter setup, PiSSA does indeed outperform both LoRA and MiLoRA, consistent with the claims made in the original PiSSA paper. Second, we find that all methods perform better in our hyperparameter setup, suggesting that our hyperparameter configuration is more effective. PiSSA configuration of hyperparameters has a combination of a small learning rate (2e-5) and a large batch size (128). This could potentially result in slow learning speed and sub-optimal performance for common PEFTs, as the optimal learning rate for common PEFTs is generally much higher than that for full finetuning (Lialin et al., 2023; Biderman et al., 2024). Third, in our hyperparameter setup, MiLoRA performs the best, outperforming LoRA and PiSSA by 1.4 and 4.3 average scores, respectively. In summary, when using an appropriate hyperparameter configuration, our proposed MiLoRA method achieves markedly superior performance compared to PiSSA.

## 6 Related Work

#### Parameter-Efficient Finetuning of LLMs.

After pretraining, LLMs are finetuned with instruction data or human preference data to adapt to different downstream tasks (Ouyang et al., 2022; Yu et al., 2023; Cui et al., 2023). Parameter-efficient finetuning (Lialin et al., 2023, PEFT) explores effective approaches to reduce the computational resources required during finetuning while managing to match the performance of full finetuning.
Previous PEFT methods can be grouped into three lines: adapter-, LoRA- and prompt-based methods. Adapter-based methods insert additional trainable feedforward submodules into each Transformer layer (Houlsby et al., 2019; Pfeiffer et al., 2021a, b). However, they introduce additional computational costs during inference. LoRA-based methods (Hu et al., 2021; Liu et al., 2024b; Zhang et al., 2023; Wang et al., 2024a) model the changes of selected linear weights as low-rank matrices. During finetuning, given a linear matrix, only two low-rank matrices are optimized while the pretrained weight matrix is frozen. Prompt-based methods (Lester et al., 2021; Li and Liang, 2021; Liu et al., 2022) add additional soft prompts to the input tokens. During training, only soft prompts are updated while the pretrained model parameters are fixed. Among these PEFT methods, the LoRA-based approaches are widely used for LLM fine-tuning because they are easy to implement and do not introduce computational overhead during inference.

#### LoRA and Its Variants

LoRA (Hu et al., 2021) reparameterizes the weight update with two trainable low-rank matrices, while freezing the pretrained weights. With this lightweight decomposition, LoRA reduces storage and task-switching overhead by sharing the pretrained models across multiple tasks. Since then, researchers have explored and proposed different LoRA variants for PEFT.
AdaLoRA (Zhang et al., 2023) and ALoRA (Liu et al., 2024d), adaptively determine the rank of LoRA module in each weight matrix according to the importance score.
The rsLoRA (Kalajdzievski, 2023) modifies the LoRA with the appropriate scaling factor to improve the performance of large ranks.
The DoRA (Liu et al., 2024b) method decomposes the pretrained weight into the magnitude and directional components, then finetunes both for better performance. Concurrent with our work, PISSA (Meng et al., 2024) and LoRA-GA Wang et al. (2024c) propose to better approximate full finetuning by only updating the principal singular components or aligning the gradients of low-rank updates to that of full finetuning.

## 7 Conclusion

In this paper, we introduce MiLoRA, a simple yet effective low-rank adaption method for LLMs. MiLoRA effectively learns on finetuning tasks while better preserving the pretrained knowledge by adapting the minor singular components of pretrained weight matrices. We investigate the effectiveness of MiLoRA on a wide range of LLM evaluation benchmarks, including commonsense reasoning, math reasoning and instruction-following, and vision-language model evaluation benchmarks. Experiment results demonstrate that MiLoRA consistently outperforms LoRA and PiSSA without sacrificing training or inference efficiency. We hope that our work will inspire future research on parameter-efficient finetuning of LLMs.

## Limitations

Due to limited computational resources, we primarily evaluate the effectiveness of MiLoRA on LLaMA-family LLMs and LLaVA-1.5, focusing on tasks such as commonsense reasoning, math reasoning, and instruction following, in line with prior work Liu et al. (2024b); Meng et al. (2024); Wu et al. (2024b). We examine and compare MiLoRA with baselines using the hyperparameter configurations provided in the previous works Hu et al. (2023); Liu et al. (2024a) instead of an exhaustive hyperparameter search for each task. We leave the exploration of MiLoRA on other tasks and other LLMs like Mistral and Gemma as future work.

## Acknowledgements

This project was supported by National Natural Science Foundation of China (No. 62306132, No. 62106138) and Guangdong Basic and Applied Basic Research Foundation. We thank the anonymous reviewers for their insightful feedback on this work.

## References

- Aghajanyan et al. (2021)
Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. 2021.
[Intrinsic dimensionality explains the effectiveness of language model fine-tuning](https://doi.org/10.18653/v1/2021.acl-long.568).
In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 7319–7328, Online. Association for Computational Linguistics.
- AI@Meta (2024)
AI@Meta. 2024.
[Llama 3 model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md).
- Biderman et al. (2024)
Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John Patrick Cunningham. 2024.
LoRA learns less and forgets less.
*Transactions on Machine Learning Research*.
Featured Certification.
- Bisk et al. (2020)
Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020.
Piqa: Reasoning about physical commonsense in natural language.
In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 7432–7439.
- Brown et al. (2020)
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, and et.al. 2020.
Language models are few-shot learners.
*arXiv preprint arXiv:2005.14165*.
- Clark et al. (2019)
Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019.
BoolQ: Exploring the surprising difficulty of natural yes/no questions.
In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2924–2936, Minneapolis, Minnesota.
- Clark et al. (2018)
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018.
Think you have solved question answering? try arc, the ai2 reasoning challenge.
*arXiv preprint arXiv:1803.05457*.
- Cobbe et al. (2021)
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021.
Training verifiers to solve math word problems.
*arXiv preprint arXiv:2110.14168*.
- Cui et al. (2023)
Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023.
Ultrafeedback: Boosting language models with high-quality feedback.
*arXiv preprint arXiv:2310.01377*.
- Dou et al. (2024)
Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang. 2024.
[Loramoe: Alleviate world knowledge forgetting in large language models via moe-style plugin](https://arxiv.org/abs/2312.09979).
*Preprint*, arXiv:2312.09979.
- Dubois et al. (2024)
Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. 2024.
Length-controlled alpacaeval: A simple way to debias automatic evaluators.
*arXiv preprint arXiv:2404.04475*.
- Goyal et al. (2017)
Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017.
Making the v in vqa matter: Elevating the role of image understanding in visual question answering.
In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 6904–6913.
- Gurari et al. (2018)
Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. 2018.
Vizwiz grand challenge: Answering visual questions from blind people.
In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3608–3617.
- Hajimolahoseini et al. (2021)
Habib Hajimolahoseini, Mehdi Rezagholizadeh, Vahid Partovinia, Marzieh S. Tahaei, Omar Mohamed Awad, and Yang Liu. 2021.
Compressing pre-trained language models using progressive low rank decomposition.
In *NeurIPS*.
- Hayou et al. (2024)
Soufiane Hayou, Nikhil Ghosh, and Bin Yu. 2024.
Lora+: Efficient low rank adaptation of large models.
*arXiv preprint arXiv:2402.12354*.
- Hendrycks et al. (2021)
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021.
Measuring mathematical problem solving with the math dataset.
*NeurIPS*.
- Houlsby et al. (2019)
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019.
Parameter-efficient transfer learning for NLP.
In *Proceedings of the 36th International Conference on Machine Learning*, volume 97, pages 2790–2799.
- Hu et al. (2021)
Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021.
Lora: Low-rank adaptation of large language models.
In *International Conference on Learning Representations*.
- Hu et al. (2023)
Zhiqiang Hu, Lei Wang, Yihuai Lan, Wanyu Xu, Ee-Peng Lim, Lidong Bing, Xing Xu, Soujanya Poria, and Roy Lee. 2023.
LLM-adapters: An adapter family for parameter-efficient fine-tuning of large language models.
In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 5254–5276, Singapore.
- Hudson and Manning (2019)
Drew A Hudson and Christopher D Manning. 2019.
Gqa: A new dataset for real-world visual reasoning and compositional question answering.
In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 6700–6709.
- Ivison et al. (2023)
Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. 2023.
Camels in a changing climate: Enhancing lm adaptation with tulu 2.
*arXiv preprint arXiv:2311.10702*.
- Jiang et al. (2023)
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023.
Mistral 7b.
*arXiv preprint arXiv:2310.06825*.
- Jiang et al. (2024)
Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. 2024.
[FollowBench: A multi-level fine-grained constraints following benchmark for large language models](https://aclanthology.org/2024.acl-long.257).
In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4667–4688, Bangkok, Thailand. Association for Computational Linguistics.
- Kalajdzievski (2023)
Damjan Kalajdzievski. 2023.
A rank stabilization scaling factor for fine-tuning with lora.
*arXiv preprint arXiv:2312.03732*.
- Kalajdzievski (2024)
Damjan Kalajdzievski. 2024.
Scaling laws for forgetting when fine-tuning large language models.
*arXiv preprint arXiv:2401.05605*.
- Kopiczko et al. (2024)
Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M Asano. 2024.
VeRA: Vector-based random matrix adaptation.
In *The Twelfth International Conference on Learning Representations*.
- Krishna et al. (2017)
Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. 2017.
Visual genome: Connecting language and vision using crowdsourced dense image annotations.
*Int. J. Comput. Vision*, 123(1):32–73.
- Lester et al. (2021)
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.
The power of scale for parameter-efficient prompt tuning.
In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 3045–3059, Online and Punta Cana, Dominican Republic.
- Li et al. (2018)
Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. 2018.
[Measuring the intrinsic dimension of objective landscapes](https://arxiv.org/abs/1804.08838).
*Preprint*, arXiv:1804.08838.
- Li and Liang (2021)
Xiang Lisa Li and Percy Liang. 2021.
Prefix-tuning: Optimizing continuous prompts for generation.
In *Proceedings of ACL*, pages 4582–4597, Online.
- Li et al. (2023a)
Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023a.
Alpacaeval: An automatic evaluator of instruction-following models.
[https://github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval).
- Li et al. (2023b)
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023b.
Evaluating object hallucination in large vision-language models.
In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 292–305, Singapore.
- Li et al. (2024)
Yixia Li, Boya Xiong, Guanhua Chen, and Yun Chen. 2024.
[Setar: Out-of-distribution detection with selective low-rank approximation](https://proceedings.neurips.cc/paper_files/paper/2024/file/8555cf308efef02adacb0f9e09bd5f10-Paper-Conference.pdf).
In *Advances in Neural Information Processing Systems*, volume 37, pages 72840–72871. Curran Associates, Inc.
- Lialin et al. (2023)
Vladislav Lialin, Vijeta Deshpande, and Anna Rumshisky. 2023.
Scaling down to scale up: A guide to parameter-efficient fine-tuning.
*arXiv preprint arXiv:2303.15647*.
- Liu et al. (2024a)
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024a.
Visual instruction tuning.
*Advances in neural information processing systems*, 36.
- Liu et al. (2024b)
Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024b.
Dora: Weight-decomposed low-rank adaptation.
*arXiv preprint arXiv:2402.09353*.
- Liu et al. (2022)
Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2022.
P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks.
In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, pages 61–68.
- Liu et al. (2024c)
Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2024c.
Mmbench: Is your multi-modal model an all-around player?
In *ECCV*.
- Liu et al. (2024d)
Zequan Liu, Jiawen Lin, Wei Zhu, and Xing Tian. 2024d.
Alora: Allocating low-rank adaptation for fine-tuning large language models.
In *Proceedings of NAACL*.
- Lu et al. (2024)
Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2024.
Learn to explain: multimodal reasoning via thought chains for science question answering.
In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, Red Hook, NY, USA.
- Mao et al. (2016)
Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and Kevin Murphy. 2016.
Generation and comprehension of unambiguous object descriptions.
In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 11–20.
- Marino et al. (2019)
Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019.
Ok-vqa: A visual question answering benchmark requiring external knowledge.
In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Meng et al. (2024)
Fanxu Meng, Zhaohui Wang, and Muhan Zhang. 2024.
Pissa: Principal singular values and singular vectors adaptation of large language models.
*arXiv preprint arXiv:2404.02948*.
- Mihaylov et al. (2018)
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018.
Can a suit of armor conduct electricity? a new dataset for open book question answering.
*arXiv preprint arXiv:1809.02789*.
- Mishra et al. (2019)
Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. 2019.
Ocr-vqa: Visual question answering by reading text in images.
In *2019 International Conference on Document Analysis and Recognition (ICDAR)*, pages 947–952.
- OpenAI (2023a)
OpenAI. 2023a.
Gpt-3.5 turbo updates.
[https://help.openai.com/en/articles/8555514-gpt-3-5-turbo-updates](https://help.openai.com/en/articles/8555514-gpt-3-5-turbo-updates).
- OpenAI (2023b)
OpenAI. 2023b.
Gpt-4 turbo in the openai api.
[https://help.openai.com/en/articles/8555510-gpt-4-turbo-in-the-openai-api](https://help.openai.com/en/articles/8555510-gpt-4-turbo-in-the-openai-api).
- Ouyang et al. (2022)
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, and et.al. 2022.
Training language models to follow instructions with human feedback.
In *Advances in Neural Information Processing Systems*, volume 35, pages 27730–27744.
- Pan et al. (2024)
Rui Pan, Xiang Liu, Shizhe Diao, Renjie Pi, Jipeng Zhang, Chi Han, and Tong Zhang. 2024.
LISA: Layerwise importance sampling for memory-efficient large language model fine-tuning.
*arXiv preprint arXiv:2403.17919*.
- Peng et al. (2023)
Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023.
Instruction tuning with gpt-4.
*arXiv preprint arXiv:2304.03277*.
- Pfeiffer et al. (2021a)
Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. 2021a.
AdapterFusion: Non-destructive task composition for transfer learning.
In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 487–503, Online.
- Pfeiffer et al. (2021b)
Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, and Sebastian Ruder. 2021b.
UNKs everywhere: Adapting multilingual language models to new scripts.
In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 10186–10203, Online and Punta Cana, Dominican Republic.
- Radford et al. (2021)
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021.
[Learning transferable visual models from natural language supervision](https://api.semanticscholar.org/CorpusID:231591445).
In *International Conference on Machine Learning*.
- Sakaguchi et al. (2021)
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021.
Winogrande: An adversarial winograd schema challenge at scale.
*Communications of the ACM*, 64(9):99–106.
- Sap et al. (2019)
Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019.
[Social IQa: Commonsense reasoning about social interactions](https://doi.org/10.18653/v1/D19-1454).
In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 4463–4473, Hong Kong, China. Association for Computational Linguistics.
- Schwenk et al. (2022)
Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. 2022.
A-okvqa: A benchmark for visual question answering using world knowledge.
In *ECCV*, page 146–162, Berlin, Heidelberg.
- Sharma et al. (2024)
Pratyusha Sharma, Jordan T. Ash, and Dipendra Misra. 2024.
[The truth is in there: Improving reasoning in language models with layer-selective rank reduction](https://openreview.net/forum?id=ozX92bu8VA).
In *The Twelfth International Conference on Learning Representations*.
- Sidorov et al. (2020)
Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. 2020.
Textcaps: A dataset for image captioning with reading comprehension.
In *Computer Vision – ECCV 2020*, pages 742–758, Cham.
- Singh et al. (2019)
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019.
Towards vqa models that can read.
In *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 8309–8318.
- Touvron et al. (2023)
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023.
Llama 2: Open foundation and fine-tuned chat models.
ArXiv preprint arXiv:2307.09288.
- Wang et al. (2024a)
Hanqing Wang, Bowen Ping, Shuo Wang, Xu Han, Yun Chen, Zhiyuan Liu, and Maosong Sun. 2024a.
[LoRA-flow: Dynamic LoRA fusion for large language models in generative tasks](https://doi.org/10.18653/v1/2024.acl-long.695).
In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12871–12882, Bangkok, Thailand. Association for Computational Linguistics.
- Wang et al. (2024b)
Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, and Zhifang Sui. 2024b.
Math-shepherd: Verify and reinforce llms step-by-step without human annotations.
*arXiv preprint arXiv:2312.08935*.
- Wang et al. (2024c)
Shaowen Wang, Linxi Yu, and Jian Li. 2024c.
Lora-ga: Low-rank adaptation with gradient approximation.
Technical report.
ArXiv 2407.05000.
- Wei et al. (2022)
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022.
Chain-of-thought prompting elicits reasoning in large language models.
*Advances in neural information processing systems*, 35:24824–24837.
- Wu et al. (2024a)
Muling Wu, Wenhao Liu, Xiaohua Wang, Tianlong Li, Changze Lv, Zixuan Ling, Jianhao Zhu, Cenyuan Zhang, Xiaoqing Zheng, and Xuanjing Huang. 2024a.
Advancing parameter efficiency in fine-tuning via representation editing.
*arXiv preprint arXiv:2402.15179*.
- Wu et al. (2024b)
Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. 2024b.
Reft: Representation finetuning for language models.
*arXiv preprint arXiv:2404.03592*.
- Yu et al. (2023)
Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023.
Metamath: Bootstrap your own mathematical questions for large language models.
*arXiv preprint arXiv:2309.12284*.
- Zellers et al. (2019)
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019.
Hellaswag: Can a machine really finish your sentence?
*arXiv preprint arXiv:1905.07830*.
- Zhang et al. (2023)
Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023.
Adaptive budget allocation for parameter-efficient fine-tuning.
In *The Eleventh International Conference on Learning Representations*.
- Zhang et al. (2024)
Renrui Zhang, Jiaming Han, Chris Liu, Aojun Zhou, Pan Lu, Yu Qiao, Hongsheng Li, and Peng Gao. 2024.
LLaMA-adapter: Efficient fine-tuning of large language models with zero-initialized attention.
In *The Twelfth International Conference on Learning Representations*.
- Zheng et al. (2023)
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023.
Judging llm-as-a-judge with mt-bench and chatbot arena.
In *Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track*.
- Zhou et al. (2023)
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023.
Instruction-following evaluation for large language models.
*arXiv preprint arXiv:2311.07911*.

## Appendix A Detailed Experiment Setups

### A.1 Similarity Metric Between Subspaces

Following Hu et al. (2021), we use the measure $\phi({\bm{A}},{\bm{B}})=\frac{\left\|{\bm{A}}^{\top}{\bm{B}}\right\|_{F}^{2}}{r}$ to measure the similarity between two column orthonormal matrices ${\bm{A}},{\bm{B}}\in\mathbb{R}^{d\times r}$, where $F$ donotes the Frobenius norm. The value of $\phi({\bm{A}},{\bm{B}})$ ranges from 0 to 1, where 1 indicates complete overlap and 0 indicates complete separation.

Given two matrices $\Delta{\bm{W}}$ and ${\bm{W}}$, we extract $r$ left singular vectors from each to form the subspace matrices for them, which are denoted as $\Delta{\bm{W}}_{s}$ and ${\bm{W}}_{s}$ , and then the subspace similarity is computed using $\phi(\Delta{\bm{W}}_{s},{\bm{W}}_{s})$.

### A.2 Our Hyperparameter Setup for LLM

Table [10](https://arxiv.org/html/2406.09044v3#A1.T10) shows our detailed hyperparameters. This setup follows LLM-adapters (Hu et al., 2023), therefore we denote it as LLM-adapters setup.

**Table 10: Our hyperparameter configuration on the commonsense reasoning (ComR), math reasoning (MathR) and instruction-following (InsF) tasks.**
| Hyperparameters | ComR | MathR | InsF |
| --- | --- | --- | --- |
| Rank r | 32 | 64 | 32 |
| $\alpha$ of LoRA | 64 | 128 | 64 |
| $\alpha$ of PiSSA/MiLoRA | 32 | 64 | 32 |
| Dropout | 0.05 |  |  |
| Optimizer | AdamW |  |  |
| LR | 3e-4 |  |  |
| LR Scheduler | Linear |  |  |
| Batch size | 16 |  |  |
| Warmup Steps | 100 |  |  |
| Epochs | 3 |  |  |
| Placement | query, key, value, MLP up, MLP down |  |  |

**Table 11: Our hyperparameter configurations applied in visual instruction tuning tasks.**
| Visual instruction tuning hyperparameters |  |
| --- | --- |
| Rank r | 128 |
| $\alpha$ of LoRA | 256 |
| $\alpha$ of PiSSA/MiLoRA | 128 |
| Dropout | 0.05 |
| Optimizer | AdamW |
| LR | 2e-4 |
| LR Scheduler | cosine |
| Batch Size | 16 |
| Warmup Ratio | 0.03 |
| Epochs | 1 |
| Placement | query, key, value, output, gate, MLP up, MLP down |

### A.3 PiSSA Hyperparameter Setup for LLM

Table [12](https://arxiv.org/html/2406.09044v3#A1.T12) shows the detailed hyperparameters from the PiSSA paper, which we denoted as PiSSA hyperparameter setup.

**Table 12: The detailed hyperparameter configurations used by PiSSAMeng et al. (2024).**
| PiSSA hyperparameters |  |
| --- | --- |
| $\alpha$ | Same as rank r |
| Dropout | 0.0 |
| Optimizer | AdamW |
| LR | 2e-5 |
| LR Scheduler | cosine |
| Batch Size | 128 |
| Warmup Ratio | 0.03 |
| Epochs | 1 |
| Placement | query, key, value, output, gate, MLP up, MLP down |

### A.4 Hyperparameter Setup for Vision-Language Model

For hyperparameters of finetuning the vision-language model, we follow the LoRA configuration provided by Liu et al. (2024a) without tuning for all PEFT methods for a fair comparison. Table [11](https://arxiv.org/html/2406.09044v3#A1.T11) shows the details.

## Appendix B More Experiment Results

### B.1 Experiments on Qwen2.5-7B

**Table 13: Math reasoning evaluation results for Qwen2.5-7B on GSM8K and MATH.**
| Method | GSM8K | MATH | Avg. |
| --- | --- | --- | --- |
| LoRA | 81.8 | 49.3 | 65.5 |
| PiSSA | 81.7 | 45.4 | 63.6 |
| MiLoRA | 85.5 | 48.7 | 67.1 |

To enhance the credibility of our experiments, we evaluate MiLoRA using Qwen2.5-7B on the MetaMathQA-100K dataset. As shown in Table [13](https://arxiv.org/html/2406.09044v3#A2.T13), MiLoRA outperforms LoRA and PiSSA by an average of 1.6 and 3.5 EM scores, respectively, demonstrating its effectiveness on the Qwen backbone. Additionally, we conduct experiments on LLMs such as LLaMA2 and LLaMA3 (Section [4.1](https://arxiv.org/html/2406.09044v3#S4.SS1)), as well as vision-language models like LLaVA-1.5 (Section [4.2](https://arxiv.org/html/2406.09044v3#S4.SS2)). Collectively, these results confirm the effectiveness of MiLoRA across diverse model architectures.

### B.2 Complementary Weighet Metrics Analysis

**Table 14: The Frobenius norm of ${\bm{U}}^{\top}{\bm{W}}{\bm{V}}$ where ${\bm{U}}$ and ${\bm{V}}$ are the left/right top $r$ singular vectors of either ${\bm{W}}$, a random matrix, or $\Delta{\bm{W}}$ of PEFTs. The weight matrices are taken from the MLP down projection of the 16th layer of LLaMA2-7B.**
| $\left\|{\bm{W}}\right\|_{F}=123.0$ | ${\bm{W}}$ | Random | LoRA | PiSSA | MiLoRA |
| --- | --- | --- | --- | --- | --- |
| $\left\|{\bm{U}}^{\top}{\bm{W}}{\bm{V}}\right\|_{F}=$ | 33.34 | 1.17 | 1.32 | 6.98 | 1.29 |
| $\left\|{\bm{U}}^{\top}\Delta{\bm{W}}{\bm{V}}\right\|_{F}=$ | - | - | 77.02 | 74.34 | 56.61 |
| Amplification factor | - | - | 58.35 | 10.65 | 43.88 |

We present additional results about the relationship between $\Delta{\bm{W}}$ and ${\bm{W}}$ in Table [14](https://arxiv.org/html/2406.09044v3#A2.T14), in which the weight matrices are taken from the MLP down projection of the 16th layer of LLaMA2-7B.

## Appendix C Broader Impacts

The MiLoRA method enhances model performance with lower training costs. Teams or users with limited computational resources can finetune large models using MiLoRA, promoting the broader application of large models across diverse groups.
However, the PEFT methods could potentially be exploited to finetune models for malicious purposes. It is essential to investigate rules and strategies to prevent the misuse of PEFT methods.