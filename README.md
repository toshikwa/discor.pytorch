# DisCor in PyTorch
This is a PyTorch implementation of DisCor[[1]](#references) and Soft Actor-Critic[[2,3]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

![walker](https://user-images.githubusercontent.com/37267851/83952299-eaf6c380-a872-11ea-8bb1-16e1d82c1dd2.gif) ![hammer](https://user-images.githubusercontent.com/37267851/84055501-e277c780-a9ef-11ea-9ba5-397b5d2d8f04.gif)


## Setup
If you are using Anaconda, first create the virtual environment.

```bash
conda create -n discor python=3.8 -y
conda activate discor
```

Then, you need to setup a MuJoCo license for your computer. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py
) for help.

Finally, you can install Python liblaries using pip.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you're using other than CUDA 10.2, you need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.

## Example

### MetaWorld

First, I trained DisCor and SAC on `hammer-v1` from MetaWorld tasks as below. Following the DisCor paper, I visualized success rate in addition to test return. These graphs correspond to Figure 7 and 16 in the paper.

```bash
python train.py --cuda --env_id hammer-v1 --config config/metaworld.yaml --num_steps 2000000 --algo discor
```

<img src="https://user-images.githubusercontent.com/37267851/84064097-0ee61080-a9fd-11ea-8afa-4d139b97bacb.png" title="graph" width=450> <img src="https://user-images.githubusercontent.com/37267851/84063868-aeef6a00-a9fc-11ea-8967-2563e2f22b6b.png" title="graph" width=450>

### Gym

I trained DisCor and SAC on `Walker2d-v2` from Gym tasks as below. A graph corresponds to Figure 17 in the paper.

```bash
python train.py --cuda --env_id Walker2d-v2 --config config/mujoco.yaml --algo discor
```

<img src="https://user-images.githubusercontent.com/37267851/83949440-c690ec00-a85e-11ea-8272-96183bdf4529.png" title="graph" width=450>


## References
[[1]](https://arxiv.org/abs/2003.07305) Kumar, Aviral, Abhishek Gupta, and Sergey Levine. "Discor: Corrective feedback in reinforcement learning via distribution correction." arXiv preprint arXiv:2003.07305 (2020).

[[2]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[3]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
