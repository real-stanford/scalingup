<h1> Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition</h1>
<div style="text-align: center;">

[Huy Ha](https://www.cs.columbia.edu/~huy/)$^1$, [Pete Florence](https://www.peteflorence.com/)$^2$,  [Shuran Song](https://shurans.github.io/)$^1$

$^1$ Columbia University, $^2$ Google DeepMind

[Project Page](https://www.cs.columbia.edu/~huy/scalingup) | [Arxiv](https://arxiv.org/abs/2307.14535) | [Video](https://www.cs.columbia.edu/~huy/scalingup/static/videos/scalingup.mp4)

<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="docs/assets/teaser.gif">

Scaling Up and Distilling Down is a framework for language-guided skill learning.
Give it a task description, and it will automatically generate rich, diverse robot trajectories, complete with success label and dense language labels.

<b>The best part?</b> It uses no expert demonstrations, manual
reward supervision, and no manual language annotation.

</div>
</div>

<br>

This repository contains code for language-guided data generation and language-conditioned diffusion policy training for [Scaling Up And Distilling Down](https://www.cs.columbia.edu/~huy/scalingup).
It has been tested on Ubuntu 18.04, 20.04 and 22.04, NVIDIA GTX 1080, NVIDIA RTX A6000, NVIDIA GeForce RTX 3080, and NVIDIA GeForce RTX 3090.

If you find this codebase useful, consider citing:

```bibtex
@inproceedings{ha2023scalingup,
      title={Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition}, 
      author={Huy Ha and Pete Florence and Shuran Song},
      year={2023},
      eprint={2307.14535},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

If you have any questions, please contact [me](https://www.cs.columbia.edu/~huy/) at `huy [at] cs [dot] columbia [dot] edu`.

**Table of Contents**

 - ⚙️[Setup](docs/setup.md)
 - 🚶[Codebase Walkthrough](docs/walkthrough.md)
   - [💡 Core Concepts](docs/walkthrough.md#-core-concepts)
      - [🪺 Nested Trajectories using Hierachical Actions and Policies](docs/walkthrough.md#-nested-trajectories-using-hierachical-actions-and-policies)
      - [🌳 Exploration Task Tree](docs/walkthrough.md#exploration-task-tree)
      - [🌈 Seeded Variation](docs/walkthrough.md#🌈-seeded-variation)
    - [🎛️ Control](docs/walkthrough.md#control)
    - [🏃‍♂️ Motion Planning](docs/walkthrough.md#️-motion-planning)
    - [🗣️ Language Model Queries](docs/walkthrough.md#language-model-queries)
      - [💿 Cache](docs/walkthrough.md#-cache)
      - [🔗 Linking LLM Modules Together](docs/walkthrough.md#-linking-llm-modules-together)
      - [🪙 Coin flips](docs/walkthrough.md#-coin-flips)
 - 🔬 [Reproducing](docs/reproduce.md) 
   - 📊 [Evaluation](docs/reproduce.md#evaluation) 
   - 🗄️ [Data Generation](docs/reproduce.md#data-generation-for-training)
   - 🧠 [Training](docs/reproduce.md#training)
- 🔭 [Extending](docs/extend.md)
    - [I want to add more](docs/extend.md#i-want-to-add-more)
      - 🤖 [Robots](docs/extend.md#robots-)
      - 🪑 [Assets](docs/extend.md#assets-)
        - 📜 [Tools \& Scripts](docs/extend.md#tools--scripts-)
      - 🌏 [Environments \& Tasks](docs/extend.md#environments--tasks-)
        - [New Simulators](docs/extend.md#new-simulators)
        - [New Tasks](docs/extend.md#new-tasks)
      - 🦙 [Language Models](docs/extend.md#language-models-)
    - 🖼️ [Figure Utilities](docs/extend.md#figure-utilities-️)
      - [Efficiency Plot](docs/extend.md#efficiency-plot)
      - [Visualizing Language-conditioned Outputs](docs/extend.md#visualizing-language-conditioned-outputs)
    - ✅ [Development Tips](docs/extend.md#development-tips-)
        - 🐉 [Hydra](docs/extend.md#hydra-)
        - 📷 [Headless Rendering](docs/extend.md#headless-rendering-)
        - 🖧 [Multi-processing](docs/extend.md#multi-processing-)
        - 👩‍👦‍👶 [Typing](docs/extend.md#typing-)
        - ⏱️ [Profiling](docs/extend.md#profiling-️)
        - 💽 [Data Format](docs/extend.md#data-format-)
        - 💾 [RAM Usage](docs/extend.md#ram-usage-)
    - 💀 [Known Issues](docs/extend.md#known-issues-)
    - ✅ [Training Tips](docs/extend.md#training-tips-)
      - [Mixed-Precision](docs/extend.md#mixed-precision)
 - 📽️ [Visualizations](docs/visualization.md)

# Acknowledgements


We would like to thank Cheng Chi, Zeyi Liu, Samir Yitzhak Gadre, Mengda Xu, Zhenjia Xu, Mandi Zhao and Dominik Bauer for their helpful feedback and fruitful discussions.


This work was supported in part by Google Research Award, NSF Award #2143601, and #2132519. 
We would like to thank Google for the UR5 robot hardware.
The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the sponsors.

## Code

 - [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/): The policy was built on top of their [Colab](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing), and our real-world evaluation code was modified from [theirs](https://github.com/columbia-ai-robotics/diffusion_policy#-demo-training-and-eval-on-a-real-robot)
 - [Mujoco Menagerie](https://github.com/deepmind/mujoco_menagerie): UR5 and RealSense models were modified from their models.
 - [Mujoco Scanned Objects](https://github.com/kevinzakka/mujoco_scanned_objects): Big shout out to [Kevin Zakka](https://kzakka.com/) for all his amazing open-source work, go give him a few stars ⭐
 - [3D UNet Implementation](https://github.com/wolny/pytorch-3dunet/) modified from [Adrian Wolny](https://adrianwolny.com/)'s implementation.
 - Support for [Fair Innovation FR5](https://robodk.com/robot/FAIR-Innovation/FR5) with the [Robotiq 852F](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) and [Weiss WSG50](https://weiss-robotics.com/servo-electric/wsg-series/) was [contributed](https://github.com/real-stanford/scalingup/pull/18) by [Yan Wang](https://wangyan-hlab.github.io/)! Go give him a few stars as well!