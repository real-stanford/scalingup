# Reproducing Scaling Up & Distilling Down

## Evaluation

### Data Generation Baselines

The general command template for evaluating a policy is 
```bash
python scalingup/inference.py evaluation.num_episodes=200 policy=$policy evaluation=$domain evaluation.start_episode=100000
```
where `$policy` can be:
  - `scalingup`: this is Ours
  - `scalingup_no_success_retry`: this baseline is like ours but without the Verify & Retry step
  - `scalingup_no_success_retry_planar`: this baseline is is like ours, but uses only planar action primitives and does not Verify & Retry

and `$domain` can be:
  - `table_top_bus_balance`
  - `catapult`
  - `bin_transport_test`
  - `mailbox`
  - `drawer`

A few key arguments are:
 - `evaluation.num_episodes`: how many evaluation episodes to run for. The paper uses 200.
 - `evaluation.start_episode`: this integer picks the seed where evaluation starts. To ensure the policy is evaluated on a seed (and therefore, a random object pose) that it wasn't trained on, make sure it is larger than the highest data generation seed. In my project, I never run data generation for more than 100000 trajectories, so I set it to `100000`.
 - `evaluation.num_processes`: how many parallel environment evaluations is used. The larger this number (up until `evaluation.num_episodes`), the faster evaluation will run at the cost of higher CPU and RAM usage.
 - `evaluation.remote`: if you're experiencing issues with multi-processing using Ray, you can use single-thread evaluation.
 - `evaluation.sampler_config.visualization_cam`: set this to `null` if you don't want to output videos. Otherwise, it will dump videos from the `front` camera by default to the run's directory. Related to this parameter are `evaluation.sampler_config.visualization_fps` and `evaluation.sampler_config.visualization_resolution`, which are self-explanatory.

> ❗Caution
> 
> For domains that have more than one task, you'll want to run more episodes to ensure each task has 200 episodes.



### Distilled Baselines

To evaluate diffusion policy approaches, use `policy=diffusion` and pass the checkpoint path `policy.path=/path/to/your/checkpoint.ckpt` to the same inference command above.
For instance,
```sh
python scalingup/inference.py evaluation.num_episodes=50000  evaluation=drawer policy=diffusion policy.path=/path/to/your/checkpoint.ckpt
```

## Training


#### Data Generation for Training

Evaluation and data generation uses the same inference code, so commands are similar.
For instance, if you want to generate 50000 trajectories using `scalingup` data generation baseline for the `drawer` task
```sh
python scalingup/inference.py evaluation.num_episodes=50000 policy=scalingup evaluation=drawer  
```
This will automatically create directory under under `scalingup/wandb/` for storing the trajectories.

> 🔭 **Future Work Opportunity**
> 
> Our approach did not make use of negative (unsuccessful data).
> To make sure data generation also dumps out unsuccessful experience to learn from, make sure `evaluation.sampler_config.dump_failed_trajectories` is `true`.
> I have already made unsuccessful attempts at modifying diffusion policy with classifier-free guidance (see `scalingup/config/algo/diffusion_suppress_negative.yaml` and `scalingup/config/algo/diffusion_classifier_free.yaml`) but maybe you can figure out how to make it work better!

The codebase supports domain randomization at data generation time.
To enable it, add `env/domain_rand_config=all`.
For example,
```
python scalingup/inference.py evaluation.num_episodes=50000 policy=scalingup evaluation=drawer env/domain_rand_config=all
```

After [running data generation](#data-generation), you can run training using the following command
```sh
python scalingup/train.py dataset_path=/path/to/drawer/dataset evaluation=drawer algo=diffusion_default
```
Here, `evaluation=drawer` tells the the trainer to evaluate the policy on the drawer domain after every epoch.


> 📘 **Info**
> 
> Training will log to [weights & biases](https://wandb.ai/site).
> If you are sweeping lots of experiments, you'll find it useful to set the tags.
> For instance, when I was running Sim2Real experiments on the transport domain using diffusion policies, I would add  `tags='[sim2real,transport,diffusion]'`