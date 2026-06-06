<a id="top"></a>

<div align="center">

# LAADAN-AC for Safer ICU-Sepsis Treatment Decisions

**Lagrangian Admissibility-Aware Deep Action-Nudging Actor-Critic**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.12.12-Research%20Code-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0%2Bcu128-ee4c2c.svg)
![Offline RL](https://img.shields.io/badge/Offline%20RL-ICU--Sepsis-1f6feb.svg)
![Safe RL](https://img.shields.io/badge/Safe%20RL-Admissibility--Aware-6f42c1.svg)

**Lagrangian Admissibility-Aware Deep Action-Nudging Actor-Critic for Safer ICU-Sepsis Treatment Decisions**



[Overview](#overview) • [Problem](#problem-statement-and-motivation) • [Methodology](#methodology) • [Run](#how-to-run-the-project) • [Limitations](#limitations) • [Future Work](#future-work) • [Citation](#license-and-citation)

</div>

---

<p align="center">
  <img src="Architecture-Diagram/LAADAN-AC-300DPI.png" width="82%" alt="LAADAN-AC project architecture">
</p>

---

## Overview

This project is an **offline deep reinforcement learning study** on the **ICU-Sepsis benchmark**.

The benchmark is **not a live clinical system**. It is a standardised Markov decision process (MDP) built from real ICU data and intended for **algorithm evaluation**, not for direct medical deployment.

The project asks one main question:

> Can a benchmark-specific offline actor-critic policy be designed so that it achieves strong survival-oriented performance **and** stays safer and more clinically plausible than simpler alternatives?

The answer supported by the final saved checkpoints is **yes, within this benchmark setting**.

The proposed model, **LAADAN-AC**, achieved the strongest overall trade-off across return, survival, and safety among the four compared agents.

---

## Project Metadata

| Field                  | Detail                                                                                                   |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| Project title          | Lagrangian Admissibility-Aware Deep Action-Nudging Actor-Critic for Safer ICU-Sepsis Treatment Decisions |
| Short name             | LAADAN-AC                                                                                                |
| Project type           |  Coursework Project                                                           |
| Main domain            | Safe offline reinforcement learning                                                                      |
| Benchmark              | ICU-Sepsis                                                                                               |
| Task                   | Sequential sepsis treatment decision-making                                                              |
| Environment type       | Tabular Markov decision process                                                                          |
| Main input             | 47-dimensional state-centre features                                                                     |
| Number of states       | 716                                                                                                      |
| Number of actions      | 25                                                                                                       |
| Evaluation horizon     | 50                                                                                                       |
| Main comparison agents | BC, CQL, VOAC, LAADAN-AC                                                                                 |
| Repository focus       | Coursework implementation of the proposed LAADAN-AC method                                               |

---

## What This Project Is About

The task is based on sepsis treatment in intensive care.

At each decision step, an agent observes a patient state and chooses one of **25 discretised treatment actions**. The goal is to maximise survival-related return over a fixed horizon.

The benchmark setting makes it possible to compare offline reinforcement learning methods under the same states, actions, transitions, rewards, expert policy, and admissibility information.

---

## Problem Statement and Motivation

Sepsis treatment is a sequential clinical decision problem. Clinicians repeatedly adjust interventions such as fluids and vasopressors under uncertainty.

Reinforcement learning is attractive here because it is built for sequential decision-making under delayed outcomes. However, the setting is also difficult:

* medical data are observational,
* unsafe actions are unacceptable,
* and policies that look good numerically can still recommend poorly supported or unrealistic actions.

The ICU-Sepsis benchmark is useful because it fixes the environment and makes algorithm comparison more reproducible.

It provides a tabular MDP with:

* **716 states**,
* **25 actions**,
* an initial-state distribution,
* transition and reward tables,
* an expert policy,
* admissibility information,
* and extra files such as `stateClusterCenters.csv` and `admissibleActions.txt`.

The core gap addressed in this coursework is the following:

* standard imitation learning may be too conservative,
* value-based offline RL may improve return but still behave implausibly,
* and a plain actor-critic may become unsafe if it is not guided away from unsupported actions.

This project therefore proposes a **project-defined** actor-critic variant that combines several safety-relevant ideas into one benchmark-specific model.

---

## Proposed Method

### Existing Algorithms Used in the Comparison

Two of the four compared agents are **pre-existing literature baselines**.

| Agent                             | Role                                                                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Behaviour Cloning (BC)**        | A standard imitation-learning baseline that learns to mimic an expert policy from state-action supervision.                          |
| **Conservative Q-Learning (CQL)** | A standard offline RL baseline that regularises Q-values conservatively to reduce over-optimistic estimates for unsupported actions. |

### Project-Defined Models

The remaining two models are **project-defined names used in this coursework**.

| Agent                                   | Role                                                                                                                 |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Vanilla Offline Actor-Critic (VOAC)** | The ablation model. It is the plain actor-critic backbone used here without the main LAADAN-specific guidance terms. |
| **LAADAN-AC**                           | The proposed model. It stands for **Lagrangian Admissibility-Aware Deep Action-Nudging Actor-Critic**.               |

---

## Why These Baselines Were Chosen

### Behaviour Cloning

BC is the simplest meaningful clinical comparator because it asks:

> What happens if the system only imitates the clinician's policy?

If a more complex RL system cannot beat imitation, its added complexity is difficult to justify.

### Conservative Q-Learning

CQL is a strong offline RL baseline because it directly targets one of offline RL’s main problems:

> value overestimation on unsupported actions.

### Vanilla Offline Actor-Critic

VOAC is the required **ablation comparator**. It asks:

> If the actor-critic backbone is kept, do the extra LAADAN safety/guidance components actually matter?

Without this ablation, any improvement by LAADAN-AC could be wrongly attributed to using actor-critic alone.

### Fair Comparison Setting

The comparison is fair because all four agents use:

* the same benchmark,
* the same 47-dimensional state-centre features,
* the same horizon,
* the same evaluation pipeline,
* and the same seed set for the main experiment.

Only the learning method changes.

---

## Benchmark and Data

The project uses the released **ICU-Sepsis benchmark** and does not require new patient data.

According to the paper and repository, the environment is a tabular MDP with:

| Benchmark property      | Value |
| ----------------------- | ----: |
| States                  |   716 |
| Actions                 |    25 |
| Reward for survival     |    +1 |
| Reward for death        |     0 |
| Terminal survival state |   714 |
| Terminal death state    |   713 |

The benchmark also includes:

* transition tables,
* reward tables,
* initial-state probabilities,
* expert policy,
* `admissibleActions.txt`,
* and `stateClusterCenters.csv`.

This coursework uses the released **continuous state cluster centres** as neural input features.

That keeps the work inside the public benchmark while still allowing a deep RL redesign.

---

## Methodology

### Shared Design

All models are evaluated on the same benchmark description.

| Setting           | Value |
| ----------------- | ----: |
| Number of states  |   716 |
| Number of actions |    25 |
| Feature dimension |    47 |
| Horizon           |    50 |

The main experiment was run over five seeds:

```text
42, 43, 44, 45, 46
```

The benchmark description and training configuration were saved in JSON so that the run can be reproduced.

---

## Model Definitions

### Behaviour Cloning

| Item      | Detail                           |
| --------- | -------------------------------- |
| Input     | 47-dimensional state features    |
| Network   | MLP                              |
| Output    | 25-way action distribution       |
| Objective | mimic the released expert policy |
| Purpose   | clinician-imitation baseline     |

### Conservative Q-Learning

| Item      | Detail                                             |
| --------- | -------------------------------------------------- |
| Input     | 47-dimensional state features                      |
| Network   | MLP Q-network                                      |
| Output    | Q-value for each of 25 actions                     |
| Objective | exact Bellman regression with conservative penalty |
| Purpose   | standard offline value-based baseline              |

### Vanilla Offline Actor-Critic

| Item    | Detail                                      |
| ------- | ------------------------------------------- |
| Input   | 47-dimensional state features               |
| Network | shared encoder + actor + two reward critics |
| Output  | stochastic policy over 25 actions           |
| Purpose | plain actor-critic ablation                 |

### LAADAN-AC

| Item    | Detail                                                     |
| ------- | ---------------------------------------------------------- |
| Input   | 47-dimensional state features                              |
| Network | shared encoder + actor + twin reward critics + cost critic |
| Purpose | proposed benchmark-specific safe offline actor-critic      |

Additional LAADAN-AC structure:

* admissibility-aware masking,
* Lagrangian cost control,
* conservative critic regularisation,
* expert-policy KL regularisation,
* smoothness-aware penalty on large action deviation.

---

## How to Run the Project

### 1. Clone the Repository with Git LFS

This repository stores the `data/` and `results/` folders using **Git Large File Storage (Git LFS)**, as the ICU-Sepsis transition table and saved model checkpoints are large files.

Install and initialise Git LFS before cloning:

```bash
git lfs install
```

Then clone the repository:

```bash
git clone https://github.com/AnnyaB/Admissibility-Aware-Offline-Actor-Critic-Learning-for-Safer-ICU-Sepsis-Treatment-Decisions.git
cd Admissibility-Aware-Offline-Actor-Critic-Learning-for-Safer-ICU-Sepsis-Treatment-Decisions
git lfs pull
```

The `git lfs pull` command downloads the real dataset and saved-result files instead of only the small pointer files.

After this, the repository should contain:

```text
data/icu_sepsis/
results/final_models/
results/train_results/
results/test_results/
scripts/
Architecture-Diagram/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Main Experiment

```bash
python run_experiments.py \
  --data-dir /path/to/data/icu_sepsis \
  --results-dir results \
  --device auto \
  --mode main
```

### 4. Expected Training Setting

The main run was executed on Kaggle with:

| Item    | Value          |
| ------- | -------------- |
| NumPy   | `2.0.2`        |
| PyTorch | `2.10.0+cu128` |
| GPU     | Tesla T4       |

CPU testing is also supported, but training is slower.

### 5. Verify the Final Selected Checkpoints

```bash
python test_final_models.py \
  --data-dir data/icu_sepsis \
  --device cpu \
  --output-json results/final_models/test_summary.json
```

### 6. What the Test Script Checks

This command:

* loads `results/final_models/bc/model.pt`,
* loads `results/final_models/cql/model.pt`,
* loads `results/final_models/voac/model.pt`,
* loads `results/final_models/laadan-ac/model.pt`,
* reconstructs their architectures from `results/run_config.json`,
* recomputes exact metrics,
* and writes the verification report to JSON.

---

## Limitations

These are the main limitations that could not be tackled due to the coursework time constraint.

| Limitation                   | Detail                                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Benchmark-only evaluation    | The project evaluates algorithms on ICU-Sepsis, not on live patients.                                                                                            |
| No clinical deployment claim | The ICU-Sepsis authors explicitly warn that the benchmark should not be used to guide medical practice directly.                                                 |
| Discrete action space        | Real treatment is continuous and more complex than 25 bins.                                                                                                      |
| State abstraction            | The model does not observe raw patient trajectories. It uses benchmark state abstractions and cluster centres.                                                   |
| No prospective validation    | Clinical usefulness would require much stronger validation, including external evaluation and medical oversight.                                                 |
| No robotics deployment       | The work is conceptually relevant to safe RL in robotics, but the current code is not a robotics controller. The aim is to extend it further to social robotics. |

---

## Future Work

Future areas of enhancement include:

* reporting full mean ± confidence intervals from the five-seed summary in the final written report,
* adding stronger offline policy evaluation,
* testing different cost budgets,
* expanding the hyperparameter study,
* comparing against additional safe offline RL methods,
* extending from discrete actions toward continuous treatment control,
* and studying whether similar admissibility-aware actor-critic ideas transfer to safety-critical robotics tasks.

---

## Future Research Vision

### JEPA-Inspired World Models for Safe Adaptive Robot Control

A further research direction is adapting **LAADAN-AC beyond the ICU-Sepsis benchmark into a more general safe robot-control algorithm for real-world and social robotics tasks**.

For the coursework, LAADAN-AC is evaluated on a fixed tabular MDP. The longer-term aim is to study whether its admissibility-aware actor-critic structure could be combined with learned world models for embodied agents that must act safely in uncertain physical and social environments.

This direction is inspired by LeCun’s proposal for autonomous machine intelligence, where intelligent agents should learn predictive world models, plan in latent representation space, and use those predictions to choose actions.

For future development, LAADAN-AC could be extended so that the actor-critic policy does not only receive a fixed benchmark state, but also uses a learned latent world model to anticipate the consequences of possible actions before selecting them.

In safer social robotics, this could mean using admissibility constraints, cost control, and expert guidance to prevent unsafe actions while a JEPA-based world model predicts likely future states.

Recent JEPA-based work, such as LeWorldModel, further supports this direction by showing how joint-embedding predictive architectures can learn compact latent world models from visual input and use them for control-related prediction and planning.

Therefore, the future research aim is not simply to improve the ICU-Sepsis benchmark result, but to investigate whether LAADAN-AC can become a broader adaptive safety framework for robotics: one that combines offline reinforcement learning, admissibility-aware action selection, Lagrangian cost control, and JEPA-inspired latent world modelling for safer real-world robot behaviour.

---

## License and Citation

### License

This project is released under the MIT License.

This means the code may be used, copied, modified, merged, published, distributed, sublicensed, and reused in future research or software projects, provided that the original copyright notice and MIT License text are included.

The ICU-Sepsis benchmark files and any third-party resources remain subject to their original providers’ terms. Users should check and follow the benchmark and dataset licences before redistributing data or derived assets.

### Citation

If this repository, code, saved checkpoints, experiment scripts, or LAADAN-AC implementation are useful in your work, please cite:

```text
Basak, R. (2026) Lagrangian Admissibility-Aware Deep Action-Nudging Actor-Critic for Safer ICU-Sepsis Treatment Decisions. Coursework Project, University of Hertfordshire. Available at: https://github.com/AnnyaB/Admissibility-Aware-Offline-Actor-Critic-Learning-for-Safer-ICU-Sepsis-Treatment-Decisions
```

---

## Medical Disclaimer

This software is for research and educational use only.

It is not a certified medical device and **must not** be used for clinical diagnosis, patient management, treatment recommendation, or treatment decisions.

Any outputs produced by this code are experimental and *may be* incorrect.

---

## References

Achiam, J., Held, D., Tamar, A. and Abbeel, P. (2017) Constrained policy optimization, in *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*, Proceedings of Machine Learning Research, 70, pp. 22-31. https://doi.org/10.48550/arXiv.1705.10528

Altman, E. (1999) *Constrained Markov Decision Processes*. Boca Raton, FL: Chapman & Hall/CRC. https://doi.org/10.1201/9781315140223

Brunke, L., Greeff, M., Hall, A.W., Yuan, Z., Zhou, S., Panerati, J. and Schoellig, A.P. (2022) Safe learning in robotics: from learning-based control to safe reinforcement learning, *Annual Review of Control, Robotics, and Autonomous Systems*, 5, pp. 411-444. https://doi.org/10.1146/annurev-control-042920-020211

Choudhary, K., Gupta, D. and Thomas, P.S. (2024) ICU-Sepsis: a benchmark MDP built from real medical data, in *Proceedings of the Reinforcement Learning Conference (RLC 2024)*. https://doi.org/10.48550/arXiv.2406.05646

Evans, L., Rhodes, A., Alhazzani, W., Antonelli, M., Coopersmith, C.M., French, C., Machado, F.R., McIntyre, L., Ostermann, M., Prescott, H.C., Schorr, C., Simpson, S., Wiersinga, W.J., Alshamsi, F., Angus, D.C., Arabi, Y., Azevedo, L., Beale, R., Beilman, G. and Levy, M.M. (2021) Surviving sepsis campaign: international guidelines for management of sepsis and septic shock 2021, *Intensive Care Medicine*, 47(11), pp. 1181-1247. https://doi.org/10.1007/s00134-021-06506-y

García, J. and Fernández, F. (2015) A comprehensive survey on safe reinforcement learning, *Journal of Machine Learning Research*, 16, pp. 1437-1480. Available at: https://www.jmlr.org/papers/v16/garcia15a.html (Accessed: 12 May 2026).

Gottesman, O., Johansson, F., Komorowski, M., Faisal, A., Sontag, D., Doshi-Velez, F. and Celi, L.A. (2019) Guidelines for reinforcement learning in healthcare, *Nature Medicine*, 25(1), pp. 16-18. https://doi.org/10.1038/s41591-018-0310-5

Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S. (2018) Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor, in *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*, Proceedings of Machine Learning Research, 80, pp. 1861-1870. https://doi.org/10.48550/arXiv.1801.01290

Harris, C.R., Millman, K.J., van der Walt, S.J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N.J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M.H., Brett, M., Haldane, A., del Río, J.F., Wiebe, M., Peterson, P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C. and Oliphant, T.E. (2020) Array programming with NumPy, *Nature*, 585, pp. 357-362. https://doi.org/10.1038/s41586-020-2649-2

Huang, Y., Cao, R. and Rahmani, A.-M. (2022) Reinforcement learning for sepsis treatment: a continuous action space solution, in *Proceedings of the 7th Machine Learning for Healthcare Conference*, Proceedings of Machine Learning Research, 182, pp. 631-647. Available at: https://proceedings.mlr.press/v182/huang22a.html (Accessed: 12 April 2026).

Hunter, J.D. (2007) Matplotlib: a 2D graphics environment, *Computing in Science & Engineering*, 9(3), pp. 90-95. https://doi.org/10.1109/MCSE.2007.55

Hussein, A., Gaber, M.M., Elyan, E. and Jayne, C. (2017) Imitation learning: a survey of learning methods, *ACM Computing Surveys*, 50(2), Article 21. https://doi.org/10.1145/3054912

icu-sepsis (2024) *The ICU-Sepsis Environment*. GitHub repository. Available at: https://github.com/icu-sepsis/icu-sepsis (Accessed: 27 March 2026).

JGraph Ltd. (2026) *diagrams.net*. Available at: https://www.diagrams.net/ (Accessed: 12 May 2026).

Jia, Y., Burden, J., Lawton, T. and Habli, I. (2020) Safe reinforcement learning for sepsis treatment, in *2020 IEEE International Conference on Healthcare Informatics (ICHI)*. IEEE, pp. 1-7. https://doi.org/10.1109/ICHI48887.2020.9374367

Jia, Y., Lawton, T., Burden, J., McDermid, J. and Habli, I. (2021) Safety-driven design of machine learning for sepsis treatment, *Journal of Biomedical Informatics*, 117, 103762. https://doi.org/10.1016/j.jbi.2021.103762

Kober, J., Bagnell, J.A. and Peters, J. (2013) Reinforcement learning in robotics: a survey, *The International Journal of Robotics Research*, 32(11), pp. 1238-1274. https://doi.org/10.1177/0278364913495721

Komorowski, M., Celi, L.A., Badawi, O., Gordon, A.C. and Faisal, A.A. (2018) The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care, *Nature Medicine*, 24(11), pp. 1716-1720. https://doi.org/10.1038/s41591-018-0213-5

Kostrikov, I., Nair, A. and Levine, S. (2022) Offline reinforcement learning with implicit Q-learning, in *Proceedings of the 10th International Conference on Learning Representations (ICLR 2022)*. Available at: https://openreview.net/forum?id=68n2s9ZJWF8 (Accessed: 12 May 2026).

Kumar, A., Zhou, A., Tucker, G. and Levine, S. (2020) Conservative Q-learning for offline reinforcement learning, in *Advances in Neural Information Processing Systems*, 33, pp. 1179-1191. https://doi.org/10.48550/arXiv.2006.04779

Lasota, P.A., Fong, T. and Shah, J.A. (2017) A survey of methods for safe human-robot interaction, *Foundations and Trends in Robotics*, 5(4), pp. 261-349. https://doi.org/10.1561/2300000052

LeCun, Y. (2022) A path towards autonomous machine intelligence, *OpenReview position paper*. Available at: https://openreview.net/pdf?id=BZ5a1r-kVsf (Accessed: 10 April 2026).

Levine, S., Kumar, A., Tucker, G. and Fu, J. (2020) Offline reinforcement learning: tutorial, review, and perspectives on open problems, *arXiv preprint arXiv:2005.01643*. https://doi.org/10.48550/arXiv.2005.01643

Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D. and Wierstra, D. (2016) Continuous control with deep reinforcement learning, in *Proceedings of the 4th International Conference on Learning Representations (ICLR 2016)*. https://doi.org/10.48550/arXiv.1509.02971

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y. and Balestriero, R. (2026) LeWorldModel: stable end-to-end joint-embedding predictive architecture from pixels, *arXiv preprint arXiv:2603.19312*. https://doi.org/10.48550/arXiv.2603.19312

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. and Chintala, S. (2019) PyTorch: an imperative style, high-performance deep learning library, in *Advances in Neural Information Processing Systems*, 32, pp. 8024-8035. https://doi.org/10.48550/arXiv.1912.01703

Pomerleau, D.A. (1989) ALVINN: an autonomous land vehicle in a neural network, in *Advances in Neural Information Processing Systems*, 1, pp. 305-313. Available at: https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html (Accessed: 20 April 2026).

Python Software Foundation (2026) *The Python Standard Library*. Available at: https://docs.python.org/3/library/ (Accessed: 22 March 2026).

Raghu, A., Komorowski, M., Celi, L.A., Szolovits, P. and Ghassemi, M. (2017) Continuous state-space models for optimal sepsis treatment, in *Proceedings of the 2nd Machine Learning for Healthcare Conference*, Proceedings of Machine Learning Research, 68, pp. 147-163. Available at: https://proceedings.mlr.press/v68/raghu17a.html (Accessed: 8 April 2026).

Ravichandar, H., Polydoros, A.S., Chernova, S. and Billard, A. (2020) Recent advances in robot learning from demonstration, *Annual Review of Control, Robotics, and Autonomous Systems*, 3, pp. 297-330. https://doi.org/10.1146/annurev-control-100819-063206

Rudd, K.E. et al. (2020) Global, regional, and national sepsis incidence and mortality, 1990-2017: analysis for the Global Burden of Disease Study, *The Lancet*, 395(10219), pp. 200-211. [https://doi.org/10.1016/S0140-6736(19)32989-7](https://doi.org/10.1016/S0140-6736%2819%2932989-7)

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) Proximal policy optimization algorithms, *arXiv preprint arXiv:1707.06347*. https://doi.org/10.48550/arXiv.1707.06347

Sutton, R.S. and Barto, A.G. (2018) *Reinforcement Learning: An Introduction*. 2nd edn. Cambridge, MA: MIT Press.

Tessler, C., Mankowitz, D.J. and Mannor, S. (2019) Reward constrained policy optimization, in *Proceedings of the 7th International Conference on Learning Representations (ICLR 2019)*. Available at: https://openreview.net/forum?id=SkfrvsA9FX (Accessed: 10 April 2026).

Tu, R., Luo, Z., Pan, C., Wang, Z., Su, J., Zhang, Y. and Wang, Y. (2025) Offline safe reinforcement learning for sepsis treatment: tackling variable-length episodes with sparse rewards, *Human-Centric Intelligent Systems*, 5, pp. 63-76. https://doi.org/10.1007/s44230-025-00093-7

Williams, R.J. (1992) Simple statistical gradient-following algorithms for connectionist reinforcement learning, *Machine Learning*, 8, pp. 229-256. https://doi.org/10.1007/BF00992696

Wu, M., Du, X., Gu, R. and Wei, J. (2021) Artificial intelligence for clinical decision support in sepsis, *Frontiers in Medicine*, 8, 665464. https://doi.org/10.3389/fmed.2021.665464

Wu, X., Li, R., He, Z., Yu, T. and Cheng, C. (2023) A value-based deep reinforcement learning model with human expertise in optimal treatment of sepsis, *npj Digital Medicine*, 6, Article 15. https://doi.org/10.1038/s41746-023-00755-5

Xu, H., Zhan, X. and Zhu, X. (2022) Constraints penalized Q-learning for safe offline reinforcement learning, in *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(8), pp. 8753-8760. https://doi.org/10.1609/aaai.v36i8.20855

Zhang, B. and Mi, Y. (2026) Safe offline reinforcement learning for sepsis treatment: a two-stage framework combining constraint-aware learning with runtime safety filtering, *Transactions on Artificial Intelligence*, 2(1), pp. 103-118. https://doi.org/10.53941/tai.2026.100007

---

<div align="center">

**[Back to top](#top)**

</div>

