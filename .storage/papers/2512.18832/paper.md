Title: From Word to World : Can Large Language Models be Implicit Text-based World Models?
ArXiv: 2512.18832
Authors: Yixia Li, Southern University of Science and Technology, Microsoft Research, Hongru Wang, University of Edinburgh, Jiahao Qiu, Princeton University, Zhenfei Yin, Oxford University, Dongdong Zhang, Cheng Qian, University of Illinois Urbana-Champaign, Zeping Li, Fudan University, Pony Ma, Mind Lab, Guanhua Chen, Heng Ji
Sections: 48
Estimated tokens: 19.4k

## Contents
- 1 Introduction
- 2 Related Works
- 3 LLM as Text-based World Models
  - 3.1 Formalization of World Models
    - Agent
    - World Model
    - Interactive Process
  - 3.2 Text-based Environments
- 4 World Model Training and Evaluation
  - Data
  - Initialization Context
  - Finetuning Models
  - Training
  - Metrics
- 5 Fidelity & Consistency
  - 5.1 Next-state Prediction Fidelity
  - 5.2 Rollout Consistency
    - Consistency Across Environments
    - How does behavior shift affect consistency?
- 6 Scalability & Robustness
  - 6.1 Data Scaling Laws for World Models
  - 6.2 Model Size Effects
  - 6.3 Beyond Memorized Environments
  - 6.4 Cross-Env Transfer via Joint Training
  - 6.5 Behavioral Coverage for Robust World Modeling
- 7 Agent Utility
  - 7.1 Can World Models Prevent Irreversible Mistakes?
  - 7.2 Synthetic Data Competes with Real
  - 7.3 Early Experience for Policy Learning
- 8 Conclusion
- References
- Appendix A Implementation Details
  - A.1 World Model Training and Evaluation
    - Environments
    - Data Sources and Sizes
    - Trajectories Collection
    - World Model Training Hyper-parameters
    - World Model Backbones
    - API Models
  - A.2 Synthetic Data Competes with Real
  - A.3 Early Experience for Policy Learning
    - Early Experience (WM-SFT)
    - Agent Warmup (Agent-SFT)
    - Reinforcement Learning (Agent-RL)
  - A.4 World Model Initialization Context
- Appendix B Task Examples and Case Studies
- Appendix C Detailed Results
- Appendix D System Prompts for Agent Trajectory Collection

## Abstract

Abstract Code: https://github.com/X1AOX1A/Word2World Agentic reinforcement learning increasingly relies on experience-driven scaling, yet real-world environments remain non-adaptive, limited in coverage, and difficult to scale. World models offer a potential way to improve learning efficiency through simulated experience, but it remains unclear whether large language models can reliably serve this role and under what conditions they meaningfully benefit agents. We study these questions in text-based environments, which provide a controlled setting to reinterpret language modeling as next-state prediction under interaction. We introduce a three-level framework for evaluating LLM-based world models: (i) fidelity and consistency, (ii) scalability and robustness, and (iii) agent utility. Across five representative environments, we find that sufficiently trained world models maintain coherent latent state, scale predictably with data and model size, and improve agent performance via action verification, synthetic trajectory generation, and warm-starting reinforcement learning. Meanwhile, these gains depend critically on behavioral coverage and environment complexity, delineating clear boundry on when world modeling effectively supports agent learning.

## 1 Introduction

Recent progress in agentic reinforcement learning (RL) increasingly hinges on experience-driven scaling: as agents improve, further gains demand larger, more diverse, and more challenging environments (Zeng et al., 2025; Zhang et al., 2025a; Tong et al., 2025). Unlike static pretraining corpora, however, experience must be collected through interaction. As agents scale, this reliance exposes a fundamental experience bottleneck: realistic environments are non-adaptive, difficult to scale, and limited in coverage, which become key limiting factors for progress in agentic RL (Wei et al., 2025; Jiang et al., 2025; Guo et al., 2025).

A key lever for alleviating this bottleneck is world modeling (Hafner et al., 2024, 2025; Zhao et al., 2025; Hu et al., 2025a). Effective learning from interaction requires agents to maintain internal models of the environment that track latent state, predict action outcomes, and learning beyond immediate observations. By internalizing environment dynamics and enabling learning from imagined interaction, world models help close the interaction loop and enable more efficient and effective learning. Meanwhile, LLM trained at massive scale via next-token prediction, exhibit strong generalization and encode rich world knowledge (Grattafiori et al., 2024; Qwen et al., 2025; Hu et al., 2025b). This naturally raises the question:

Can large language models serve as effective world models,
thereby improving agents to learn from experience?

While prior work has explored LLMs as simulators, experience generators, or planning interfaces (Chen et al., 2025; Li et al., 2025b; Wu et al., 2025; Gu et al., 2025; Wang et al., 2025; He et al., 2025), it remains unclear *how* to learn a world model and *when* it is reliable enough to improve downstream agents. A useful world model must go beyond locally plausible text, maintaining coherent state over time, remaining robust to distribution shift, and providing measurable utility. To address these questions in a controlled setting, we focus on text-based environments as a unifying interface between language modeling and world modeling. This abstraction preserves core challenges of agentic-environment interaction while reframing the objective from next-token prediction to next-state prediction under a fixed interaction protocol.

Through this lens, we formalize a three-level framework for characterizing world modeling capabilities in agent learning.

Studying five representative text-based environments, our analysis yields three main findings:
(i) LLMs can function as reliable world models: they exhibit internal latent dynamics that support in-context world modeling, and supervised fine-tuning substantially improves short-term predictive fidelity and enables consistent long-horizon rollouts in well-structured domains.
(ii) The development of reliable world models requires systematic scaling of model capacity and data volume with environment complexity, and maintaining robustness to distribution shift through broad behavioral coverage and diverse environment exposure.
(iii) Fidelity world models provide practical benefits to agents by enabling verification of high-stakes actions to prevent irreversible failures, and by generating synthetic trajectories and warm-starting reinforcement learning to improve learning efficiency and effectiveness.

Taken together, these findings illuminate both the promise and the limits of LLM-based world models in text environments. From words to worlds, from next-token to next-state prediction, we provide an empirical foundation for treating LLMs as general-purpose world models for agentic learning and chart a path toward domains beyond text.

## 2 Related Works

Large language models have recently been explored as world models across a variety of text-based and structured settings. Prior efforts in world modeling largely focus on predicting environment dynamics through structured or discrete state representations. Patch-based approaches prompt LLMs to estimate state deltas in ByteSized32 (Wang et al., 2024; Yang et al., 2024), while in web navigation, systems such as WMA (Chae et al., 2025) and RLVR-World (Wu et al., 2025) reason over updates to the Accessibility Tree. Other lines of work adopt closed-form prediction schemes where the model outputs predefined symbolic labels or categories, including preconditions and effects in cooking environments (Xie et al., 2024), disaster impact ratings (Li et al., 2025a), or classifier-head predictions trained on LLM embeddings (Yang et al., 2025). Although these methods illustrate the utility of structured prediction for specific settings, they generally depend on environment-specific abstractions and a fixed output space tailored to particular domains. In contrast, we formulate world modeling as a multi-turn natural language simulation task, where the LLM generates next-state transitions in free text, enabling more general and compositional interaction patterns.

Regarding model adaptation, much prior work employs zero-shot or few-shot prompting (Wang et al., 2024; Yang et al., 2024; Li et al., 2025a; Zuo et al., 2025) or attaches lightweight classifier heads for closed-form prediction (Yang et al., 2025). While such settings highlight the latent capabilities of LLMs, they often yield limited accuracy and constrains their applicability in downstream tasks.
Moving beyond prompting-based adaptations, we finetune LLMs on large-scale multi-turn interaction trajectories to better internalize environment dynamics over extended horizons.

Prior evaluation efforts largely center on single-step prediction accuracy in limited environments and domains (Wang et al., 2024; Xie et al., 2024; Chae et al., 2025; Li et al., 2025a) , and rarely examines long-horizon consistency or compounding errors—factors that are critical for using world models as reliable simulators. Consequently, it remains an open question whether LLM-based world models can produce coherent multi-step trajectories that are executable in real environments. To address this gap, we conduct a systematic evaluation across five representative environments, measuring not only one-step fidelity but also rollout stability, WM-to-Real transfer, and generalization across agents, environments, and scales.

## 3 LLM as Text-based World Models

### 3.1 Formalization of World Models

We formalize the interaction between an agent and a text-based world model as a multi-turn language-based decision process, where both perception and action are represented in natural language.

#### Agent

A text-based agent $\mathcal{A}$ operates in a ReAct style (Yao et al., 2023b), yielding a simple, unified interface where each step involves internal reasoning and external action.
Formally, the agent is defined as:

$$ $\mathcal{A}:\{S_{0},(T_{i},A_{i},S_{i})_{i=1}^{n-1}\}\rightarrow(T_{n},A_{n}),$ (1) $$

where $S_{i}$ denotes the textual observation (or environment response) at step $i$,
$T_{i}$ represents the agent’s internal reasoning trace,
and $A_{i}$ denotes the explicit action expressed in natural language.

#### World Model

The environment or its surrogate world model $\mathcal{W}$ defines the complementary mapping:

$$ $\mathcal{W}:\{S_{0},(A_{i},S_{i}^{\prime})_{i=1}^{n-1},A_{n}\}\rightarrow(S_{n}^{\prime},R_{n}^{\prime}),$ (2) $$

where $S_{n}^{\prime}$ denotes the next state predicted by the world model, and $R_{n}^{\prime}\in\{0,1\}$ is a binary reward indicating task success or termination.
A value of $R_{n}^{\prime}=1$ corresponds to a successful completion, while $R_{n}^{\prime}=0$ denotes either an unfinished or failure state (e.g., triggering validation at the wrong time).
Through these textual transitions, the world model functions as an implicit next-state predictor of environment dynamics.
This capability can be realized through in-context learning, where the model leverages few-shot examples of state transitions in its prompt, or through supervised fine-tuning on trajectory data to learn the underlying dynamics.

Note that in practice, text-based environments are inherently POMDPs (Partially Observable Markov Decision Processes): the true environment state is richer than what is described to the agent in text. For example, in ALFWorld a room may contain objects and spatial details that are never mentioned (e.g., what is inside a closed drawer), yet these hidden factors matter for predicting how the world evolves.
Thus, although the agent only receives a partial view of the initial state $S_{0}$, the world model can be initialized with a more complete context such as full environment configurations or randomized setups, allowing it to better approximate the latent dynamics of the environment.

#### Interactive Process

Together, the agent and world model form an iterative process:

$$ $S_{n}^{\prime},R_{n}^{\prime}=\mathcal{W}\big(\mathcal{A}(S_{0},(T_{i},A_{i},S_{i}^{\prime})_{i=1}^{n-1})\big),$ (3) $$

which unrolls into a multi-turn textual trajectory generated within the world model:

$$ $\tau_{\text{wm}}=\{S_{0},T_{1},A_{1},S_{1}^{\prime},\dots,T_{T},A_{T},S_{T}^{\prime}\}.$ (4) $$

Correspondingly, the real environment produces the trajectory

$$ $\tau_{\text{real}}=\{S_{0},T_{1},A_{1},S_{1},\dots,T_{T},A_{T},S_{T}\},$ (5) $$

which serves as the reference for evaluating the fidelity and consistency of $\mathcal{W}$.

By formulating text-based environments as multi-turn interactive processes, the world model can be prompted with few-shot examplars or trained on real trajectories $\tau_{\text{real}}$ to predict next-state transitions. This formulation enables $\mathcal{W}$ to capture long-horizon dependencies and cumulative effects across interaction steps.
While prior works (Wang et al., 2024; Xie et al., 2024; Yang et al., 2025) primarily focus on next-state prediction accuracy, we explicitly model, train and evaluate the world model’s long-horizon consistency, which is critical for applications such as data synthesis, test-time simulator, and model-based reinforcement learning.

### 3.2 Text-based Environments

To examine the range of knowledge and dynamics required for text-based world modeling in a broad way, we adopt five representative environments spanning both structured and open-ended settings. The structured environments ALFWorld (Shridhar et al., 2021), SciWorld (Wang et al., 2022), and TextWorld (Côté et al., 2018) feature bounded state spaces.
They provide deterministic or rule-governed transitions grounded in embodied, scientific, or narrative regularities.
In contrast, the open-ended environments WebShop (Yao et al., 2023a) and StableToolBench (Guo et al., 2025) exhibit broad, compositional, and open-world dynamics, with diverse entities and flexible task formulations that require stronger generalization beyond fixed schemas. Table [5](#A1.T5) in Appendix [A.1](#A1.SS1) summarizes these environments and their key characteristics, with examples in Appendix [B](#A2). Together, these settings provide a comprehensive and diversified testbed for evaluating language models as text-based world simulators.

## 4 World Model Training and Evaluation

We summarize the world model training and evaluation setup and defer full implementation details to Appendix [A.1](#A1.SS1).
Unless otherwise specified, all experiments follow the default settings described in this section.

#### Data

We collect interaction trajectories using GPT-4o as the behavior policy. To match environment complexity (see Section [6.1](#S6.SS1)), we gather 40K trajectories each for ALFWorld, SciWorld, and TextWorld, and 70K for WebShop. We retain both successful and failed episodes to broaden behavioral coverage for world model training.(^1^11The success/failure mixture is induced by GPT-4o’s native success rate without additional filtering.) For StableToolBench, we use the public single-turn dataset with 160K samples.

#### Initialization Context

For ALFWorld and SciWorld, the world model is provided with full initial state descriptions (see Figures [8](#A1.F8) and [9](#A1.F9) in Appendix [A.4](#A1.SS4)).
In contrast, TextWorld does not expose complete initial states, and WebShop/StableToolBench are inherently partially observable. This setting places greater demands on history-based state tracking and prior knowledge to infer unobserved state variables.

#### Finetuning Models

We use Qwen2.5-7B (base) and Llama-3.1-8B (base) as backbone models for text-based world modeling. A comparison across different model sizes is provided in Section [6.2](#S6.SS2).

#### Training

Each trajectory is formatted as a multi-turn dialogue of alternating agent actions and environment responses (see Eq. [5](#S3.E5)). During supervised fine-tuning, the world model predicts the next environment response conditioned on the dialogue history and the current action.

#### Metrics

We evaluate world models along two dimensions: one-step prediction fidelity and multi-step rollout consistency.
Fidelity. We compute exact-match (EM) accuracy by conditioning on a real trajectory prefix $\{S_{0},(A_{i},S_{i})_{i=1}^{n-1},A_{n}\}$ and predicting the next state and reward $(S_{n}^{\prime},R_{n}^{\prime})$. A prediction is correct if $(S_{n}^{\prime},R_{n}^{\prime})$ matches the ground-truth $(S_{n},R_{n})$. For TextWorld, EM is a conservative lower bound since multiple surface forms can describe the same underlying state. For StableToolBench, whose outputs are highly open-ended, we additionally report word-level F1.
Consistency. We report: (1) Real: success rate in the real environment; (2) WM: success rate inside the world model; (3) W2R: success rate when replaying WM actions in the real environment; and (4) Consistency Ratio: $\text{CR}=\text{W2R}/\text{Real}$, where higher values indicate better long-horizon transfer (CR may exceed 1 when world model rollouts are more successful than real world).

## 5 Fidelity & Consistency

### 5.1 Next-state Prediction Fidelity

Table [1](#S5.T1) demonstrates that pretrained LLMs exhibit meaningful in-context world modeling ability.
Models such as Gemini-2.5-flash and Claude-sonnet-4.5 achieve strong next-state prediction in structured environments like ALFWorld and SciWorld, where a handful of demonstrations provides substantial improvements (e.g., Claude rises from 56.83 to 73.08 accuracy on SciWorld with only three examples).
This suggests that contemporary LLMs encode latent knowledge of environment dynamics and can rapidly adapt their transition rules with minimal supervision.
However, these capabilities do not fully transfer to open-ended settings such as WebShop, where few-shot prompting plateaus around mid-50s, indicating that implicit world knowledge alone is insufficient for generating unconstrained, context-dependent state updates.

Supervised fine-tuning yields substantial improvements.
Open-source models trained directly on transition trajectories achieve 99%/98% accuracy on ALFWorld and SciWorld and reach 49% F1 on StableToolBench.
These results indicate that robust world modeling requires dynamics-aligned training: prompting alone cannot capture the full diversity of transition patterns, whereas supervised fine-tuning enables even relatively small models to internalize them effectively.

**Table 1: Next-state prediction EM accuracy (%) of prompt-based and finetuned models across five environments. AW, SW, TW, WS and STB denote ALFWorld, SciWorld, TextWorld, WebShop and StableToolBench, respectively. STB${}_{\text{F1}}$ denotes the word-level F1 score for StableToolBench, given its open-ended output space.**
| Environment | AW | SW | TW | WS | STB | STB${}_{\text{F1}}$ |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot |  |  |  |  |  |  |
| GPT-4o-mini | $45.20$ | $40.68$ | $0.36$ | $56.59$ | $0.00$ | $13.94$ |
| GPT-4o | $44.45$ | $45.78$ | $7.86$ | $58.20$ | $0.00$ | $11.88$ |
| GPT-4-turbo | $42.64$ | $34.14$ | $0.00$ | $52.45$ | $0.00$ | $12.64$ |
| GPT-4.1 | $43.56$ | $35.65$ | $0.00$ | $58.07$ | $0.00$ | $12.83$ |
| GPT-5 | $35.09$ | $13.06$ | $9.20$ | $46.12$ | $0.00$ | $8.02$ |
| Gemini-2.5-flash | $50.00$ | $44.81$ | $3.51$ | $57.64$ | $0.00$ | $8.74$ |
| Claude-sonnet-4.5 | $64.73$ | $56.83$ | $17.70$ | $58.80$ | $0.00$ | $11.36$ |
| Few-shot (3 shot) |  |  |  |  |  |  |
| GPT-4o-mini | $63.79$ | $56.26$ | $11.43$ | $61.93$ | $0.00$ | $13.44$ |
| GPT-4o | $56.88$ | $48.98$ | $14.11$ | $64.62$ | $0.00$ | $11.08$ |
| GPT-4-turbo | $62.56$ | $50.08$ | $11.66$ | $62.76$ | $0.00$ | $10.72$ |
| GPT-4.1 | $63.37$ | $51.56$ | $13.39$ | $64.23$ | $0.00$ | $10.33$ |
| GPT-5 | $67.13$ | $49.44$ | $44.27$ | $65.90$ | $0.00$ | $6.28$ |
| Gemini-2.5-flash | $61.85$ | $61.20$ | $40.35$ | $66.09$ | $0.00$ | $8.47$ |
| Claude-sonnet-4.5 | $77.04$ | $73.08$ | $49.12$ | $56.65$ | $0.00$ | $13.11$ |
| SFT |  |  |  |  |  |  |
| Qwen2.5-7B | $99.87$ | $98.60$ | $70.60$ | $79.05$ | $48.90$ | $79.15$ |
| Llama3.1-8B | $99.71$ | $98.64$ | $70.45$ | $77.24$ | $49.25$ | $78.97$ |

**Table 2: Task success rate (%) of different agents across four environments. “Real”, “WM”, and “W2R” denote the success rate under real environment, world model, and world model-to-real execution. The last column reports the consistency ratio (CR=W2R/Real), with higher values (darker green color) indicating better rollout fidelity.**
| Agent | ALFWorld | SciWorld | TextWorld | WebShop |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Real | WM | W2R | CR | Real | WM | W2R | CR | Real | WM | W2R | CR | Real | WM | W2R | CR |  |
| Qwen2.5-7B WorldModel |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GPT-4o-mini | $7.69$ | $7.69$ | $7.69$ | 1.00 | $12.64$ | $12.04$ | $8.90$ | 0.70 | $97.44$ | $100.00$ | $69.36$ | 0.71 | $5.99$ | $4.85$ | $0.97$ | 0.16 |
| GPT-4o | $58.00$ | $55.90$ | $57.44$ | 0.99 | $34.97$ | $37.63$ | $31.44$ | 0.90 | $98.84$ | $100.00$ | $96.53$ | 0.98 | $29.36$ | $17.43$ | $16.51$ | 0.56 |
| GPT-4-turbo | $74.21$ | $62.56$ | $64.62$ | 0.87 | $36.79$ | $50.00$ | $36.60$ | 0.99 | $100.00$ | $99.42$ | $98.84$ | 0.99 | $17.73$ | $14.89$ | $11.70$ | 0.66 |
| GPT-4.1 | $67.20$ | $68.56$ | $69.59$ | 1.04 | $43.41$ | $45.79$ | $46.32$ | 1.07 | $100.00$ | $100.00$ | $100.00$ | 1.00 | $21.14$ | $12.22$ | $12.22$ | 0.58 |
| GPT-5 | $91.00$ | $84.62$ | $86.67$ | 0.95 | $68.21$ | $64.10$ | $61.03$ | 0.89 | $100.00$ | $100.00$ | $100.00$ | 1.00 | $51.00$ | $33.03$ | $31.19$ | 0.61 |
| Gemini-2.5-flash | $50.50$ | $51.79$ | $52.31$ | 1.04 | $56.00$ | $39.49$ | $45.64$ | 0.82 | $100.00$ | $100.00$ | $76.30$ | 0.76 | $25.00$ | $21.10$ | $18.35$ | 0.73 |
| Claude-sonnet-4.5 | $82.00$ | $76.00$ | $76.00$ | 0.93 | $66.00$ | $45.64$ | $57.95$ | 0.88 | $100.00$ | $100.00$ | $100.00$ | 1.00 | $61.00$ | $49.00$ | $50.00$ | 0.82 |
| Average | $61.51$ | $58.16$ | $59.19$ | 0.96 | $45.43$ | $42.10$ | $41.13$ | 0.91 | $99.47$ | $99.92$ | $91.58$ | 0.92 | $30.17$ | $21.79$ | $20.13$ | 0.67 |
| Llama3.1-8B WorldModel |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GPT-4o-mini | $7.69$ | $9.74$ | $9.74$ | 1.27 | $12.64$ | $10.78$ | $8.21$ | 0.65 | $97.44$ | $92.48$ | $57.80$ | 0.59 | $5.99$ | $2.75$ | $0.00$ | 0.00 |
| GPT-4o | $58.00$ | $58.46$ | $56.92$ | 0.98 | $34.97$ | $37.63$ | $32.99$ | 0.94 | $98.84$ | $97.11$ | $90.17$ | 0.91 | $29.36$ | $23.81$ | $22.62$ | 0.77 |
| GPT-4-turbo | $74.21$ | $67.53$ | $67.01$ | 0.90 | $36.79$ | $52.31$ | $44.10$ | 1.20 | $100.00$ | $97.69$ | $93.06$ | 0.93 | $17.73$ | $25.47$ | $17.92$ | 1.01 |
| GPT-4.1 | $67.20$ | $68.72$ | $68.21$ | 1.02 | $43.41$ | $45.13$ | $35.38$ | 0.82 | $100.00$ | $98.27$ | $94.22$ | 0.94 | $21.14$ | $19.27$ | $17.43$ | 0.82 |
| GPT-5 | $91.00$ | $82.56$ | $81.54$ | 0.90 | $68.21$ | $63.07$ | $57.44$ | 0.84 | $100.00$ | $98.84$ | $94.80$ | 0.95 | $51.00$ | $31.19$ | $30.28$ | 0.59 |
| Gemini-2.5-flash | $50.50$ | $53.33$ | $53.33$ | 1.06 | $56.00$ | $57.95$ | $52.31$ | 0.93 | $100.00$ | $99.42$ | $93.06$ | 0.93 | $25.00$ | $22.02$ | $17.43$ | 0.70 |
| Claude-sonnet-4.5 | $82.00$ | $84.00$ | $84.00$ | 1.02 | $66.00$ | $58.46$ | $53.33$ | 0.81 | $100.00$ | $93.33$ | $90.00$ | 0.90 | $61.00$ | $60.00$ | $55.00$ | 0.90 |
| Average | $61.51$ | $60.62$ | $60.11$ | 0.98 | $45.43$ | $46.48$ | $30.52$ | 0.89 | $99.47$ | $96.73$ | $87.59$ | 0.88 | $30.17$ | $26.36$ | $22.95$ | 0.76 |

### 5.2 Rollout Consistency

A reliable world model requires not only high single-step prediction accuracy, but more critically, the ability to maintain consistency during extended interactions with agents. We examine two key dimensions: (1) whether small local errors compound into significant failures over long-horizon rollouts, and (2) whether the world model generalizes across different agent behaviors beyond its training distribution. Table [2](#S5.T2) reports consistency metrics across four environments and multiple agents; StableToolBench is omitted due to its single-turn nature.

#### Consistency Across Environments

World models largely preserve single-step fidelity in long-horizon rollouts, especially in structured environments. In ALFWorld, SciWorld, and TextWorld, the fine-tuned Qwen2.5 world model attains high consistency ratios of 96%, 91%, and 92%, indicating that multi-step trajectories generated within the world model remain executable when transferred to the real environment.
WebShop, however, exhibits lower consistency (typically below 80%), primarily due to its open-ended nature and diverse search results that the world model struggles to simulate accurately.
This error can be substantially mitigated by grounding model rollouts with real observations. When the rollout is initialized with real search results, the consistency with GPT-4o agent increases dramatically from 56% to nearly 100%, demonstrating that partial real-environment anchoring effectively reduces simulation drift.

#### How does behavior shift affect consistency?

Beyond environment-specific factors, world model consistency also depends on how well agent behaviors match the training distribution. Lower-capacity agents such as GPT-4o-mini yield consistency ratios frequently below 70%, whereas stronger agents like GPT-4.1, GPT-5, and Claude reliably exceed 90%. This disparity stems from weaker agents taking actions misaligned with task objectives, causing their trajectories to drift outside the training distribution. In contrast, higher-capacity agents preserve goal-directed behavior that aligns with the expert policy (GPT-4o) used for trajectories sampling, enabling higher consistency.
These results highlight the importance of diversifying training trajectories rather than relying solely on a single strong agent, as further discussed in Section [6.5](#S6.SS5).

## 6 Scalability & Robustness

### 6.1 Data Scaling Laws for World Models

Figure: Figure 2: Next-state prediction accuracy under varying training data sizes on Qwen2.5-7B. Structured settings saturate with modest data (~20K), whereas open-ended settings continue to benefit from larger datasets. Note. We apply a nonlinear y-axis transform $f(y)=100-20\log_{10}(\max(100-y,0.01)+1)$ to better reveal growth trends.
Refer to caption: 2512.18832v2/x1.png

Figure: Figure 3: Next-state prediction accuracy on Qwen2.5 family. Smaller models (~1.5B) capture structured dynamics effectively, whereas more complex settings benefit markedly from increased model capacity.
Refer to caption: 2512.18832v2/x2.png

To investigate how world model performance scales with data, we vary training trajectories from 1K to 160K and evaluate single-step accuracy. As shown in Figure [2](#S6.F2), structured environments (ALFWorld, SciWorld, TextWorld) improve rapidly and saturate around 20K trajectories, consistent with their low-entropy, rule-driven dynamics. In contrast, open-ended environments scale more gradually: WebShop benefits from additional data up to roughly 70K trajectories, while StableToolBench shows no saturation at 160K samples due to long-tail linguistic variation and highly compositional API behaviors. These results indicate that world modeling exhibits environment-dependent scaling: structured environments are highly data-efficient, whereas open-ended domains require substantially larger datasets.

### 6.2 Model Size Effects

We next analyze how model capacity shapes world model performance (Figure [3](#S6.F3)). Mirroring data-scaling trends, model size interacts strongly with environment complexity. In structured environments, performance saturates quickly: 1.5B models already capture core transition dynamics, with further scaling yielding only marginal improvements. In open-ended environments, however, capacity matters substantially. Smaller models struggle to represent rich linguistic variability and compositional tool usage, whereas larger models offer steady accuracy gains. Together with the data-scaling results, these findings indicate that success in open-ended world modeling requires both extensive trajectories and sufficient model capacity to internalize long-tailed, high-entropy dynamics.

### 6.3 Beyond Memorized Environments

Figure: Figure 4: Task success rate (%) in ALFWorld under different OOD settings. Success rate averaged over different agents, with full results provided in Table [10](#A3.T10) of Appendix [C](#A3). World models maintain strong performance even when layouts or room types change.
Refer to caption: 2512.18832v2/x3.png

A central question in world model design is how well they generalize across unseen settings. Using ALFWorld as a representative case, we analyze two out-of-distribution test splits following the original environment settings (Shridhar et al., 2021): OOD-Seen, which keeps the room type but alters the layout, and OOD-Unseen, which introduces entirely new room types or unseen layout configurations.
As shown in Figure [4](#S6.F4), the world models maintain success rates closely aligned with the real environment across both OOD settings even when the spatial configuration shifts or novel room types appear. These results indicate that the LLM world model captures transferable transition dynamics rather than memorizing specific layouts, demonstrating strong robustness to structural variations in environment state space.

### 6.4 Cross-Env Transfer via Joint Training

Figure: Figure 5: Next-state prediction accuracy under mixed and separate training on Qwen2.5-7B, with 1K samples per environment. We begin by mixing structured environments (ALFWorld, SciWorld, TextWorld) and then progressively incorporate open-ended environments (WebShop, StableToolBench), yielding the Mix3, Mix4, and Mix5 settings.
Refer to caption: 2512.18832v2/x4.png

Training world models in isolation often limits their ability to generalize beyond a single environment, motivating us to investigate whether jointly training on multiple environments can yield transferable gains. We therefore evaluate three mixed-training configurations: Mix3 (ALFWorld, SciWorld, TextWorld), Mix4 (with WebShop), and Mix5 (with StableToolBench), allocating 1K trajectories per environment to match the data budget of individually trained models. As shown in Figure [5](#S6.F5), mixed training consistently accelerates learning and improves final accuracy, with particularly strong gains in TextWorld and WebShop, suggesting that the model effectively internalizes and reuses shared physical, procedural, and narrative dynamics across tasks. The exception is StableToolBench, whose schema-centric, single-turn structure is underrepresented in the mixture, causing separately trained model to outperform. Overall, these results show that mixed data provides stable positive gains and, importantly, enables practical deployments where a single world model can robustly serve multiple environments.

### 6.5 Behavioral Coverage for Robust World Modeling

**Table 3: Task success rate (%) in SciWorld under different training data compositions. “Single Agent Traj” uses only 4K GPT-4o trajectories for training, whereas “Mix Agent Traj” combines trajectories from ID agents, with 1K trajectories from each.**
| Agent |  | Single Agent Traj | Mix Agent Traj |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Real | WM | W2R | CR | WM | W2R | CR |  |
| Qwen3-235B | 24.00 | $24.00$ | $18.00$ | 0.75 | $26.00$ | $18.00$ | 0.75 |
| GPT-4o | 34.97 | $32.31$ | $26.67$ | 0.76 | $32.31$ | $26.67$ | 0.76 |
| GPT-5 | 68.21 | $55.38$ | $59.49$ | 0.87 | $69.74$ | $60.00$ | 0.88 |
| Claude-sonnet-4.5 | 66.00 | $40.51$ | $57.44$ | 0.87 | $52.31$ | $49.74$ | 0.75 |
| ID Agent Average | 48.30 | $38.05$ | $40.40$ | 0.81 | $45.09$ | $38.60$ | 0.79 |
| GPT-4o-mini | 12.64 | $5.64$ | $6.15$ | 0.49 | $13.39$ | $10.26$ | 0.81 |
| GPT-4-turbo | 36.79 | $32.31$ | $38.97$ | 1.06 | $51.28$ | $42.56$ | 1.16 |
| GPT-4.1 | 43.41 | $28.72$ | $36.41$ | 0.84 | $52.31$ | $36.41$ | 0.84 |
| Gemini-2.5-flash | 56.00 | $36.92$ | $51.79$ | 0.92 | $56.92$ | $45.64$ | 0.82 |
| OOD Agent Average | 37.21 | $25.90$ | $33.33$ | 0.83 | $43.48$ | $33.72$ | 0.91 |

As behavior shifts reduce consistency, we ask whether broader behavioral coverage improves generalization. We train a world model on mixed-agent trajectories and compare it to a GPT-4o-only baseline. Table [3](#S6.T3) shows marked OOD gains for weaker agents: GPT-4o-mini’s consistency ratio rises from 0.49 to 0.81, and GPT-4-turbo also improves. This indicates that expert-only trajectories are insufficient under distribution shift; incorporating diverse agent behaviors is crucial for improving generalization and rollout stability.

## 7 Agent Utility

### 7.1 Can World Models Prevent Irreversible Mistakes?

**Table 4: Task success rate (%) of different agents in WebShop with varying numbers of max pre-execution verification attempts using the world model. The numbers in parentheses indicate the improvement over the baseline without verification.**
| Agent | 0 | 2 | 4 | 10 | 50 |
| --- | --- | --- | --- | --- | --- |
| GPT-4o-mini | 5.99 | 7.50 (+1.51) | 7.55 (+1.56) | 7.59 (+1.60) | 7.59 (+1.60) |
| GPT-4o | 29.36 | 32.41 (+3.05) | 33.94 (+4.58) | 34.86 (+5.50) | 36.70 (+7.34) |
| GPT-4-turbo | 17.73 | 33.33 (+15.60) | 27.05 (+9.32) | 29.37 (+11.64) | 25.60 (+7.87) |
| GPT-4.1 | 21.14 | 23.59 (+2.45) | 23.59 (+2.45) | 23.08 (+1.94) | 25.13 (+3.99) |
| GPT-5 | 51.00 | 53.27 (+2.27) | 53.77 (+2.77) | 53.27 (+2.77) | 51.50 (+0.50) |
| Gemini-2.5-flash | 25.00 | 31.00 (+6.00) | 29.50 (+4.50) | 28.00 (+3.00) | 27.50 (+1.50) |
| Claude-sonnet-4.5 | 61.00 | 62.00 (+1.00) | 65.00 (+4.00) | 64.00 (+3.00) | 62.00 (+1.00) |

In real-world decision-making, some actions are irreversible and costly, creating a safety bottleneck: a single mistaken commitment can end an episode or cause unrecoverable loss. This motivates using world models as a *rewindable imagined world* to evaluate high-stakes actions before execution. WebShop exemplifies this setting: once the agent checks out, the episode ends and errors cannot be undone. We therefore use the world model as a lightweight pre-execution verifier. Before committing to checkout, the agent simulates the outcome; it executes the action only when the prediction indicates success, otherwise it continues interacting with the environment. We vary the verification budget (0, 2, 4, 10, 50).

As shown in Table [4](#S7.T4), verification improves success rates for all agents, with the largest gains for medium-capacity models. However, returns are not monotonic, since repeated verification changes the trajectory context and shifts the agent’s action distribution, inducing distribution shift that can weaken alignment between imagined and real outcomes. In practice, moderate budgets (e.g., 2–10 checks) provide the best trade-off, reducing irreversible failures without destabilizing behavior.

### 7.2 Synthetic Data Competes with Real

When real interaction is expensive, slow, or constrained, agents face an experience bottleneck. A world model can potentially alleviate this bottleneck by synthesizing trajectories that substitute for a portion of real experience. To examine this, we collect 1,000 successful trajectories from either the real environment or the world model(^2^22Implementation details in Appendix [A.2](#A1.SS2)), and construct four SFT datasets for Qwen2.5-7B-Instruct: Real 1K, Syn 1K, 0.5K+0.5K, and a 1K+1K mixture, all trained under identical procedures.
Figure [6](#S7.F6) shows that world model–generated trajectories are highly competitive with real data. In SciWorld, Syn 1K matches Real 1K, while the 1K+1K mixture outperforms either source alone. In WebShop, synthetic data remains similarly effective, and mixed regimes yield the most stable gains. Overall, these results suggest that synthetic experience can reduce reliance on real-environment interaction, providing an alternative pathway for scaling agent learning when real experience is limited.

Figure: Figure 6: Task success rate (%) of Qwen2.5-7B-Instruct SFT trained agents with different data synthesis strategies in SciWorld and WebShop.
Refer to caption: 2512.18832v2/x5.png

### 7.3 Early Experience for Policy Learning

Recent work (Zhang et al., 2025b) suggests that exposing a model to environment dynamics before explicit policy learning can provide a useful inductive bias: anticipating consequences may reduce unguided exploration and stabilize early RL. To study this in our text-based decision environments, we compare (1) standard Agent-SFT $\rightarrow$ RL baseline; and (2) world-model warmup pipeline (WM-SFT $\rightarrow$ Agent-SFT $\rightarrow$ RL), where the agent is first exposed to environment dynamics with same objective as world model training(^3^33Implementation details in Appendix [A.3](#A1.SS3)).

Figure [7](#S7.F7) indicates that early experience delivers consistent gains on both ALFWorld and SciWorld. By exposing the agent to environment dynamics before policy learning, early experience stabilizes RL training, reducing failures driven by incorrect commonsense priors, and ultimately yields higher final success rates than the baseline. Overall, early experience provides a promising direction for improving learning effectiveness.

## 8 Conclusion

This work set out to investigate a simple yet far-reaching question: can the paradigm that enables large language models to model words also enable them to model worlds, and in turn support more effective agent learning from experience? Using text-based environments as a controlled testbed, we recast world modeling as multi-turn next-state prediction under interaction, and introduce a systematic framework for evaluating fidelity, scalability, and agent utility.

Our results provide strong evidence that LLMs can serve as implicit text-based world models. When trained with dynamics-aligned supervision at sufficient scale and coverage, they maintain coherent latent state over extended horizons and yield tangible benefits for downstream agents, including safer decision making, scalable experience generation, and improved learning efficiency. At the same time, these gains are not universal: robustness depends critically on behavioral coverage, distributional alignment, and environment complexity, delineating concrete regimes in which world modeling meaningfully supports agent learning.

Taken together, these findings establish an empirical foundation for treating LLMs not merely as sequence predictors, but as learned simulators of interactive worlds. By bridging next-token prediction with next-state modeling, this work points toward a unifying view of language models as world models for agents—and opens the door to extending these ideas beyond text to richer, multimodal, and embodied domains.

*

## References

- Chae et al. [2025]
H. Chae, N. Kim, K. T. iunn Ong, M. Gwak, G. Song, J. Kim, S. Kim, D. Lee, and J. Yeo.
Web agents with world models: Learning and leveraging environment dynamics in web navigation, 2025.
URL [https://arxiv.org/abs/2410.13232](https://arxiv.org/abs/2410.13232).
- Chen et al. [2025]
Z. Chen, Z. Zhao, K. Zhang, B. Liu, Q. Qi, Y. Wu, T. Kalluri, S. Cao, Y. Xiong, H. Tong, H. Yao, H. Li, J. Zhu, X. Li, D. Song, B. Li, J. Weston, and D. Huynh.
Scaling agent learning via experience synthesis, 2025.
URL [https://arxiv.org/abs/2511.03773](https://arxiv.org/abs/2511.03773).
- Côté et al. [2018]
M.-A. Côté, A. Kádár, X. Yuan, B. Kybartas, T. Barnes, E. Fine, J. Moore, R. Y. Tao, M. Hausknecht, L. E. Asri, M. Adada, W. Tay, and A. Trischler.
Textworld: A learning environment for text-based games.
*CoRR*, abs/1806.11532, 2018.
- Grattafiori et al. [2024]
A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Roziere, B. Biron, B. Tang, B. Chern, C. Caucheteux, C. Nayak, and C. B. et al.
The llama 3 herd of models, 2024.
URL [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783).
- Gu et al. [2025]
Y. Gu, K. Zhang, Y. Ning, B. Zheng, B. Gou, T. Xue, C. Chang, S. Srivastava, Y. Xie, P. Qi, H. Sun, and Y. Su.
Is your llm secretly a world model of the internet? model-based planning for web agents, 2025.
URL [https://arxiv.org/abs/2411.06559](https://arxiv.org/abs/2411.06559).
- Guo et al. [2025]
Z. Guo, S. Cheng, H. Wang, S. Liang, Y. Qin, P. Li, Z. Liu, M. Sun, and Y. Liu.
Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models, 2025.
URL [https://arxiv.org/abs/2403.07714](https://arxiv.org/abs/2403.07714).
- Hafner et al. [2024]
D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap.
Mastering diverse domains through world models, 2024.
URL [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).
- Hafner et al. [2025]
D. Hafner, W. Yan, and T. Lillicrap.
Training agents inside of scalable world models, 2025.
URL [https://arxiv.org/abs/2509.24527](https://arxiv.org/abs/2509.24527).
- He et al. [2025]
H. He, Y. Zhang, L. Lin, Z. Xu, and L. Pan.
Pre-trained video generative models as world simulators.
In *ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling*, 2025.
URL [https://openreview.net/forum?id=oTYF8WUadL](https://openreview.net/forum?id=oTYF8WUadL).
- Hu et al. [2025a]
M. Hu, T. Chen, Y. Zou, Y. Lei, Q. Chen, M. Li, Q. Liang, Y. Mu, H. Zhang, W. Shao, and P. Luo.
Text2world: Benchmarking large language models for symbolic world model generation.
In *ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling*, 2025a.
URL [https://openreview.net/forum?id=dIQNOxuBay](https://openreview.net/forum?id=dIQNOxuBay).
- Hu et al. [2025b]
Z. Hu, J. Lian, Z. Xiao, S. Zhang, T. Wang, N. J. Yuan, X. Xie, and H. Xiong.
Unveiling the learning mind of language models: A cognitive framework and empirical study.
*arXiv preprint arXiv:2506.13464*, 2025b.
- Jiang et al. [2025]
D. Jiang, Y. Lu, Z. Li, Z. Lyu, P. Nie, H. Wang, A. Su, H. Chen, K. Zou, C. Du, T. Pang, and W. Chen.
Verltool: Towards holistic agentic reinforcement learning with tool use, 2025.
URL [https://arxiv.org/abs/2509.01055](https://arxiv.org/abs/2509.01055).
- Li et al. [2025a]
L. Li, D. Li, Z. Ou, X. Xu, J. Liu, Z. Ma, R. Yu, and M. Deng.
Llms as world models: Data-driven and human-centered pre-event simulation for disaster impact assessment, 2025a.
URL [https://arxiv.org/abs/2506.06355](https://arxiv.org/abs/2506.06355).
- Li et al. [2025b]
Y. Li, H. A. Inan, X. Yue, W.-N. Chen, L. Wutschitz, J. Kulkarni, R. Poovendran, R. Sim, and S. Rajmohan.
Simulating environments with reasoning models for agent training, 2025b.
URL [https://arxiv.org/abs/2511.01824](https://arxiv.org/abs/2511.01824).
- Qwen et al. [2025]
Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu.
Qwen2.5 technical report, 2025.
URL [https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115).
- Shridhar et al. [2021]
M. Shridhar, X. Yuan, M.-A. Côté, Y. Bisk, A. Trischler, and M. Hausknecht.
Alfworld: Aligning text and embodied environments for interactive learning, 2021.
URL [https://arxiv.org/abs/2010.03768](https://arxiv.org/abs/2010.03768).
- Tong et al. [2025]
J. Tong, J. Tang, H. Li, Y. Mou, M. Zhang, J. Zhao, Y. Wen, F. Song, J. Zhan, Y. Lu, C. Tao, Z. Guo, J. Yu, T. Cheng, Z. Xi, C. Jiang, Z. Yin, Y. Zheng, W. Ge, G. Chen, T. Gui, X. Qiu, Q. Zhang, and X. Huang.
Game-rl: Synthesizing multimodal verifiable game data to boost vlms’ general reasoning, 2025.
URL [https://arxiv.org/abs/2505.13886](https://arxiv.org/abs/2505.13886).
- Wang et al. [2022]
R. Wang, P. Jansen, M.-A. Côté, and P. Ammanabrolu.
Scienceworld: Is your agent smarter than a 5th grader?, 2022.
URL [https://arxiv.org/abs/2203.07540](https://arxiv.org/abs/2203.07540).
- Wang et al. [2024]
R. Wang, G. Todd, Z. Xiao, X. Yuan, M.-A. Côté, P. Clark, and P. Jansen.
Can language models serve as text-based world simulators?
In L.-W. Ku, A. Martins, and V. Srikumar, editors, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 1–17, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.
[10.18653/v1/2024.acl-short.1](https:/doi.org/10.18653/v1/2024.acl-short.1).
URL [https://aclanthology.org/2024.acl-short.1/](https://aclanthology.org/2024.acl-short.1/).
- Wang et al. [2025]
S. Wang, Z. Fei, Q. Cheng, S. Zhang, P. Cai, J. Fu, and X. Qiu.
World modeling makes a better planner: Dual preference optimization for embodied task planning.
In *ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling*, 2025.
URL [https://openreview.net/forum?id=HCXshVxxdg](https://openreview.net/forum?id=HCXshVxxdg).
- Wei et al. [2025]
Z. Wei, W. Yao, Y. Liu, W. Zhang, Q. Lu, L. Qiu, C. Yu, P. Xu, C. Zhang, B. Yin, H. Yun, and L. Li.
Webagent-r1: Training web agents via end-to-end multi-turn reinforcement learning, 2025.
URL [https://arxiv.org/abs/2505.16421](https://arxiv.org/abs/2505.16421).
- Wu et al. [2025]
J. Wu, S. Yin, N. Feng, and M. Long.
Rlvr-world: Training world models with reinforcement learning, 2025.
URL [https://arxiv.org/abs/2505.13934](https://arxiv.org/abs/2505.13934).
- Xi et al. [2024]
Z. Xi, Y. Ding, W. Chen, B. Hong, H. Guo, J. Wang, D. Yang, C. Liao, X. Guo, W. He, S. Gao, L. Chen, R. Zheng, Y. Zou, T. Gui, Q. Zhang, X. Qiu, X. Huang, Z. Wu, and Y.-G. Jiang.
Agentgym: Evolving large language model-based agents across diverse environments, 2024.
- Xi et al. [2025]
Z. Xi, J. Huang, C. Liao, B. Huang, H. Guo, J. Liu, R. Zheng, J. Ye, J. Zhang, W. Chen, W. He, Y. Ding, G. Li, Z. Chen, Z. Du, X. Yao, Y. Xu, J. Chen, T. Gui, Z. Wu, Q. Zhang, X. Huang, and Y.-G. Jiang.
Agentgym-rl: Training llm agents for long-horizon decision making through multi-turn reinforcement learning, 2025.
URL [https://arxiv.org/abs/2509.08755](https://arxiv.org/abs/2509.08755).
- Xie et al. [2024]
K. Xie, I. Yang, J. Gunerli, and M. Riedl.
Making large language models into world models with precondition and effect knowledge, 2024.
- Yang et al. [2024]
C. Yang, X. Wang, J. Jiang, Q. Zhang, and X. Huang.
Evaluating world models with llm for decision making, 2024.
URL [https://arxiv.org/abs/2411.08794](https://arxiv.org/abs/2411.08794).
- Yang et al. [2025]
C. Yang, X. Wang, Q. Zhang, Q. Jiang, and X. Huang.
Efficient integration of external knowledge to LLM-based world models via retrieval-augmented generation and reinforcement learning.
In C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, editors, *Findings of the Association for Computational Linguistics: EMNLP 2025*, pages 9484–9501, Suzhou, China, Nov. 2025. Association for Computational Linguistics.
ISBN 979-8-89176-335-7.
[10.18653/v1/2025.findings-emnlp.504](https:/doi.org/10.18653/v1/2025.findings-emnlp.504).
URL [https://aclanthology.org/2025.findings-emnlp.504/](https://aclanthology.org/2025.findings-emnlp.504/).
- Yao et al. [2023a]
S. Yao, H. Chen, J. Yang, and K. Narasimhan.
Webshop: Towards scalable real-world web interaction with grounded language agents, 2023a.
URL [https://arxiv.org/abs/2207.01206](https://arxiv.org/abs/2207.01206).
- Yao et al. [2023b]
S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao.
React: Synergizing reasoning and acting in language models, 2023b.
URL [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629).
- Zeng et al. [2025]
Z. Zeng, H. Ivison, Y. Wang, L. Yuan, S. S. Li, Z. Ye, S. Li, J. He, R. Zhou, T. Chen, C. Zhao, Y. Tsvetkov, S. S. Du, N. Jaques, H. Peng, P. W. Koh, and H. Hajishirzi.
Rlve: Scaling up reinforcement learning for language models with adaptive verifiable environments, 2025.
URL [https://arxiv.org/abs/2511.07317](https://arxiv.org/abs/2511.07317).
- Zhang et al. [2025a]
J. Zhang, Y. Peng, F. Kong, C. Yang, Y. Wu, Z. Yu, J. Xiang, J. Ruan, J. Wang, M. Song, H. Liu, X. Tang, B. Liu, C. Wu, and Y. Luo.
Autoenv: Automated environments for measuring cross-environment agent learning, 2025a.
URL [https://arxiv.org/abs/2511.19304](https://arxiv.org/abs/2511.19304).
- Zhang et al. [2025b]
K. Zhang, X. Chen, B. Liu, T. Xue, Z. Liao, Z. Liu, X. Wang, Y. Ning, Z. Chen, X. Fu, J. Xie, Y. Sun, B. Gou, Q. Qi, Z. Meng, J. Yang, N. Zhang, X. Li, A. Shah, D. Huynh, H. Li, Z. Yang, S. Cao, L. Jang, S. Zhou, J. Zhu, H. Sun, J. Weston, Y. Su, and Y. Wu.
Agent learning via early experience, 2025b.
URL [https://arxiv.org/abs/2510.08558](https://arxiv.org/abs/2510.08558).
- Zhao et al. [2025]
Y. Zhao, A. Scannell, Y. Hou, T. Cui, L. Chen, D. Büchler, A. Solin, J. Kannala, and J. Pajarinen.
Generalist world model pre-training for efficient reinforcement learning.
In *ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling*, 2025.
URL [https://openreview.net/forum?id=WtJnrr4BGO](https://openreview.net/forum?id=WtJnrr4BGO).
- Zheng et al. [2024]
Y. Zheng, R. Zhang, J. Zhang, Y. Ye, Z. Luo, Z. Feng, and Y. Ma.
Llamafactory: Unified efficient fine-tuning of 100+ language models.
In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)*, Bangkok, Thailand, 2024. Association for Computational Linguistics.
URL [http://arxiv.org/abs/2403.13372](http://arxiv.org/abs/2403.13372).
- Zuo et al. [2025]
D. Zuo, Z. CHEN, C. Zhou, Y. Guo, X. He, and M. Gong.
RADI: LLMs as world models for robotic action decomposition and imagination.
In *ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling*, 2025.
URL [https://openreview.net/forum?id=cPo2iS6lwP](https://openreview.net/forum?id=cPo2iS6lwP).

## Appendix A Implementation Details

### A.1 World Model Training and Evaluation

#### Environments

We evaluate five text-based environments, including ALFWorld [Shridhar et al., 2021], SciWorld [Wang et al., 2022], TextWorld [Côté et al., 2018], WebShop [Yao et al., 2023a], and StableToolBench [Guo et al., 2025]. Table [5](#A1.T5) summarizes these environments along four dimensions: the nature of the environment, the abilities required of an agent, the form of the underlying world state, and the modeling capabilities demanded of a world model.

**Table 5: Summary of the five text-based environments used in our paper, highlighting the knowledge demands placed on both agents and world models. Task examples are provided in Figures [14](#A2.F14)–[18](#A2.F18) in Appendix [B](#A2).**
| Environment | Description | Required Agent Ability | World Model State | Required World Model Ability |
| --- | --- | --- | --- | --- |
| ALFWorld [Shridhar et al., 2021] | Embodied environment where agents accomplish household tasks by issuing text-based commands. | Spatial and physical commonsense, reasoning about containers and locations, and multi-step executions. | Room layout with hundreds of container–object combinations, agent inventory, and task progression. | Track physical configurations, maintain object relations, and predict stable multi-step state transitions. |
| SciWorld [Wang et al., 2022] | Text-based interactivate laboratory environment involving simplified physics & chemistry experiments. | Scientific concepts, causal reasoning, experiment planning, hypothesis testing with outcome evaluation. | Ten interconnected labs with $\sim$200 materials, intermediate substance states, and experiment progress. | Scientific dynamics modeling, physical reasoning, chemical simulation, experiment progress estimation. |
| TextWorld [Côté et al., 2018] | Text-based open-world environment supporting exploration, interaction, and diverse quest-like tasks. | Environment understanding, open-ended task planning, temporal tracking, and structured exploration. | Multiple connected rooms with $\sim$10 objects, exploration and discovery status, and task advancement. | Long-horizon state prediction, symbolic transition feedback, and exploration progress estimation. |
| WebShop [Yao et al., 2023a] | Simulated shopping website where agents search, browse, and shop through multi-step interactions. | Goal decomposition, product evaluation, and robust reasoning over diverse semi-structured attributes. | Metadata for over 1M product attributes, search-query items surface, item details, and cart states. | Simulation of search engines, multi-step web navigation, product attributes, and constraint satisfaction. |
| StableToolBench [Guo et al., 2025] | API-based tool-use environment requiring schema adherence and structured output generation. | Doc understanding, symbolic reasoning, and executing schema-compliant action sequences. | Over 10K API tools, input/output schemas, intermediate tool-call states, and execution context. | Symbolic world state simulation, doc understanding, schema constraint satisfaction, structured generation. |

#### Data Sources and Sizes

For ALFWorld, SciWorld and WebShop, we follow the data splits provided in AgentGym(^4^44[https://github.com/WooooDyy/AgentGym](https://github.com/WooooDyy/AgentGym)) [Xi et al., 2024].
For TextWorld, we follow the official TextWorld repository(^5^55[https://github.com/microsoft/TextWorld](https://github.com/microsoft/TextWorld)) to generate game files and randomly split into 2.5K training games and 200 test games.
For StableToolBench, we filtered the StableToolBench MirrorAPI dataset(^6^66[https://huggingface.co/datasets/stabletoolbench/MirrorAPI-Training](https://huggingface.co/datasets/stabletoolbench/MirrorAPI-Training)) and remove samples with errors or incomplete information, and use 160K API pairs for training and 2K pairs for testing. The data sizes for different environments are summarized in Table [6](#A1.T6).

#### Trajectories Collection

We utilize the AgentGym [Xi et al., 2024] framework to collect long-horizon interaction trajectories using GPT-4o
as the agent across four interactive environments: ALFWorld, SciWorld, TextWorld, and WebShop. We maintain consistent system prompts (Appendix [D](#A4)), interaction protocols, environment configurations as in AgentGym.
The sampling temperature is set to 1.0 with Top-p of 1.0, and a maximum of 50 interaction turns per trajectory.
System prompts used for trajectory collection are provided in Figure [19](#A4.F19) to [23](#A4.F23).
Ultimately, we collect 40K trajectories each for ALFWorld, SciWorld, and TextWorld, and 70K trajectories for WebShop on their respective training sets, as summarized in Table [6](#A1.T6).

**Table 6: Training data sizes for different environments. StableToolBench only contains single-turn training data without interactive trajectories.**
| Environment | Train Games | Test Games | Trajectories |
| --- | --- | --- | --- |
| ALFWorld | 2420 | 200 | 40K |
| SciWorld | 2120 | 200 | 40K |
| TextWorld | 2500 | 200 | 40K |
| WebShop | 3930 | 200 | 70K |
| StableToolBench | 160K | 2000 | None |

#### World Model Training Hyper-parameters

We utilize LLaMa-Facotry(^7^77[https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)) [Zheng et al., 2024] for SFT training of LLM-based world models. The training parameters are summarized in Table [7](#A1.T7). Parameters unspecified in the table follow the default settings of LLaMA-Factory.
Training data size varies across different environments, as detailed in Table [6](#A1.T6) if not otherwise specified.
Experiments are conducted on 4xH100-80GB GPUs.

**Table 7: SFT hyper-parameters for training LLM-based world models.**
| Parameter | Value |
| --- | --- |
| Global Train Batch Size | 128 |
| Learning Rate | 1.0e-5 |
| Number of Training Epochs | 5 |
| LR Scheduler Type | Constant with Warmup |
| Warmup Steps | 10 |
| BF16 | True |
| Max Gradient Norm | 100 |

#### World Model Backbones

We utilize Qwen2.5-7B [Qwen et al., 2025] and Llama3.1-8B [Grattafiori et al., 2024] as the primary backbone for LLM-based world models. To study the impact of model scale, we train Qwen2.5 models of four sizes: 0.5B, 1.5B, 3B, and 7B parameters. The specific model checkpoints used are as follows:

**Table 8: Model checkpoints used for world model training.**
| Model | Checkpoint URL |
| --- | --- |
| Qwen2.5-7B | [https://huggingface.co/Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) |
| Qwen2.5-3B | [https://huggingface.co/Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) |
| Qwen2.5-1.5B | [https://huggingface.co/Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |
| Qwen2.5-0.5B | [https://huggingface.co/Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| Llama3.1-8B | [https://huggingface.co/meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |

#### API Models

We list the API models and their versions used in paper in Table [9](#A1.T9).

**Table 9: API models and versions used for evaluations.**
| Model | Version |
| --- | --- |
| GPT-4o-mini | gpt-4o-mini-2024-07-18 |
| GPT-4o | gpt-4o-2024-11-20 |
| GPT-4-turbo | gpt-4-turbo-2024-04-09 |
| GPT-4.1 | gpt-4.1-2025-04-14 |
| GPT-5 | gpt-5-2025-08-07 |
| Gemini-2.5-flash | gemini-2.5-flash |
| Claude-sonnet-4.5 | claude-sonnet-4-5-20250929 |
| Qwen3-235B | qwen3-235b-a22b-instruct-2507 |

### A.2 Synthetic Data Competes with Real

To compare the quality of world-model–synthesized trajectories with those from the real environment, we construct matched SFT datasets using 1,000 successful trajectories collected from (i) the real environment and (ii) the world model. To control for the behavior policy used during data collection, both datasets are generated by the same agent: a Qwen2.5-7B-Instruct policy trained via direct RL (i.e., without any SFT). This design avoids reusing the world-model training policy (GPT-4o) as the collector, thereby reducing the risk that the world model “self-replays” trajectories through the behavior policy. For world-model rollouts, trajectory success is determined by the model’s own predicted outcome.

### A.3 Early Experience for Policy Learning

#### Early Experience (WM-SFT)

To provide early dynamics exposure before policy learning, we warm-start the agent with a *world-model style* supervised objective (Eq. [2](#S3.E2)): predicting the next environment response and termination signal conditioned on the dialogue history and the current action.
We use the same data sources described in Appendix [A.1](#A1.SS1) and sample 1,000 trajectories to construct the WM-SFT dataset.
Training follows the same SFT hyper-parameters as world model training (Table [7](#A1.T7)).
For the baseline without early experience, this stage is skipped.

#### Agent Warmup (Agent-SFT)

After WM-SFT, we perform a standard policy warmup stage by supervised fine-tuning on real-environment trajectories collected in Appendix [A.1](#A1.SS1).
Specifically, we sample 1,000 trajectories and fine-tune the agent to generate its next turn (reasoning trace and action; Eq. [1](#S3.E1)) from the interaction history.
We use the same SFT hyper-parameters as in Table [7](#A1.T7).

#### Reinforcement Learning (Agent-RL)

We utilize the AgentGymRL framework(^8^88[https://github.com/WooooDyy/AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL)) [Xi et al., 2025] to run GRPO training for agent policy training, with the suggested hyper-parameters as suggested in the paper. The command is as follows:

> python3 -m verl.agent_trainer.main_ppo \ algorithm.adv_estimator=grpo \ algorithm.rounds_ctrl.type=fixed \ algorithm.rounds_ctrl.rounds=20 \ data.train_file=${DATA_FILE} \ data.train_batch_size=16 \ data.max_prompt_length=1024 \ data.max_response_length=4096 \ actor_rollout_ref.agentgym.task_name=${TASK_NAME} \ actor_rollout_ref.agentgym.env_addr=${ENV_ADDR} \ actor_rollout_ref.agentgym.timeout=600 \ actor_rollout_ref.model.path=${MODEL_PATH} \ actor_rollout_ref.actor.use_kl_loss=True \ actor_rollout_ref.actor.kl_loss_coef=0.001 \ actor_rollout_ref.actor.kl_loss_type=low_var_kl \ actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \ actor_rollout_ref.rollout.n=8 \ actor_rollout_ref.rollout.max_model_len=32768 \ actor_rollout_ref.rollout.max_tokens=200 \ actor_rollout_ref.rollout.tensor_model_parallel_size=1 \ actor_rollout_ref.actor.ppo_epochs=1 \ actor_rollout_ref.actor.optim.lr=1e-6 \ actor_rollout_ref.actor.ppo_mini_batch_size=8 \ actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \ algorithm.kl_ctrl.kl_coef=0.001 \ trainer.default_local_dir=’outputs/ckpt/${trainer.experiment_name}’ \ trainer.project_name="agentgym" \ trainer.experiment_name="${EXPERIMENT_NAME}" \ trainer.save_freq=10 \ trainer.total_epochs=10 \ trainer.n_gpus_per_node=4

### A.4 World Model Initialization Context

In ALFWorld and SciWorld, each game instance involves random initialization of the environment. For example, in ALFWorld, the positions and contents of objects within rooms vary, while in SciWorld, the connectivity of houses changes with each initialization. Consequently, even for humans, accurately predicting the next state of the environment based solely on task descriptions is challenging.
Similar to RAWM [Yang et al., 2025], we include the initial state information of the environment for the world model’s predictions. This design aligns with practical applications where the world model is used with knowledge of the initial environment state.
In data synthesis scenarios, such random states can be sampled through similar random generation methods, enhancing the diversity and generalization capabilities of the world model.
Examples of initial state information are provided in Figure [8](#A1.F8) and [9](#A1.F9).
While TextWorld lacks full initial state information due to environment limits. WebShop and StableToolBench are inherently open environments where comprehensive initial state information cannot be provided, so they also do not include initial state information.

Figure: Figure 8: Initialization Context Example of ALFWorld

Figure: Figure 9: Initialization Context Example of SciWorld

Figure: Figure 10: Initialization Context Example of SciWorld (Continued)

Figure: Figure 11: Initialization Context Example of SciWorld (Continued)

Figure: Figure 12: Initialization Context Example of SciWorld (Continued)

Figure: Figure 13: Initialization Context Example of SciWorld (Continued)

## Appendix B Task Examples and Case Studies

In this section, we provide task examples and case studies on world model across five environments.

Figure: Figure 14: Task Example and Case Study of StableToolBench

Figure: Figure 15: Task Example and Case Study of ALFWorld

Figure: Figure 16: Task Example and Case Study of SciWorld

Figure: Figure 17: Task Example and Case Study of TextWorld.

Figure: Figure 18: Task Example and Case Study of WebShop

## Appendix C Detailed Results

We provide detailed results in Table [10](#A3.T10) for the OOD generalization of world models.

**Table 10: Task success rate (%) in ALFWorld under different OOD settings. “OOD-Seen” indicates the same room with different layout as training. “OOD-Unseen” indicates tasks containing room types or environment layouts never seen during training.**
| Agent | OOD - Seen | OOD - Unseen |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Real | WM | W2R | CR | Real | WM | W2R | CR |  |
| Qwen2.5-7B WorldModel |  |  |  |  |  |  |  |  |
| GPT-4o-mini | $6.75$ | $7.10$ | $7.10$ | 1.05 | $4.03$ | $4.67$ | $5.33$ | 1.32 |
| GPT-4o | $52.10$ | $43.79$ | $45.56$ | 0.87 | $52.00$ | $44.67$ | $44.67$ | 0.86 |
| GPT-4-turbo | $65.00$ | $51.23$ | $52.47$ | 0.81 | $74.50$ | $62.42$ | $62.42$ | 0.84 |
| GPT-4.1 | $53.37$ | $56.80$ | $57.40$ | 1.08 | $64.19$ | $65.33$ | $64.67$ | 1.01 |
| GPT-5 | $71.60$ | $69.23$ | $71.01$ | 0.99 | $74.00$ | $76.00$ | $76.67$ | 1.04 |
| Gemini-2.5-flash | $39.05$ | $40.83$ | $41.42$ | 1.06 | $51.35$ | $48.67$ | $49.33$ | 0.96 |
| Claude-sonnet-4.5 | $87.00$ | $72.00$ | $79.00$ | 0.91 | $76.04$ | $76.00$ | $79.00$ | 1.04 |
| Average | $53.55$ | $48.71$ | $50.57$ | 0.94 | $56.59$ | $53.97$ | $54.58$ | 0.96 |
| Llama3.1-8B WorldModel |  |  |  |  |  |  |  |  |
| GPT-4o-mini | $6.75$ | $8.88$ | $8.88$ | 1.32 | $4.03$ | $2.67$ | $2.67$ | 0.66 |
| GPT-4o | $52.10$ | $48.52$ | $47.93$ | 0.92 | $52.00$ | $49.33$ | $49.33$ | 0.95 |
| GPT-4-turbo | $65.00$ | $56.52$ | $55.90$ | 0.86 | $74.50$ | $62.16$ | $62.16$ | 0.83 |
| GPT-4.1 | $53.37$ | $56.21$ | $55.62$ | 1.04 | $64.19$ | $60.67$ | $60.00$ | 0.93 |
| GPT-5 | $71.60$ | $69.82$ | $69.23$ | 0.97 | $74.00$ | $74.00$ | $73.33$ | 0.99 |
| Gemini-2.5-flash | $39.05$ | $42.60$ | $42.60$ | 1.09 | $51.35$ | $46.00$ | $45.33$ | 0.88 |
| Claude-sonnet-4.5 | $87.00$ | $78.00$ | $84.00$ | 0.97 | $76.04$ | $81.00$ | $78.00$ | 1.03 |
| Average | $53.55$ | $51.51$ | $52.02$ | 0.97 | $56.59$ | $53.69$ | $52.97$ | 0.94 |

## Appendix D System Prompts for Agent Trajectory Collection

Figure: Figure 19: Agent System Prompt for ALFWorld Trajectory Collection

Figure: Figure 20: Agent System Prompt for WebShop Trajectory Collection

Figure: Figure 21: Agent System Prompt for SciWorld Trajectory Collection

Figure: Figure 22: Agent System Prompt for TextWorld Trajectory Collection

Figure: Figure 23: Agent System Prompt for StableToolBench Trajectory Collection