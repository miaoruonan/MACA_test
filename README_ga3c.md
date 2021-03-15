`This` is the training code for:

**Journal Version:** M. Everett, Y. Chen, and J. P. How, "Collision Avoidance in Pedestrian-Rich Environments with Deep Reinforcement Learning", in review, [Link to Paper](https://arxiv.org/abs/1910.11689)

**Conference Version:** M. Everett, Y. Chen, and J. P. How, "Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018. [Link to Paper](https://arxiv.org/abs/1805.01956), [Link to Video](https://www.youtube.com/watch?v=XHoXkWLhwYQ)

The [gym environment code](https://github.com/mit-acl/gym-collision-avoidance) is included as a submodule.

### Install

Grab the code from github, initialize submodules, install dependencies and src code
```bash
# Clone either through SSH or HTTPS (MIT-ACL users should use GitLab origin)
git clone --recursive git@github.com:mit-acl/rl_collision_avoidance.git

cd rl_collision_avoidance
./install.sh
```

There are some moderately large (10s of MB) checkpoint files containing network weights that are stored in this repo as Git LFS files. They should automatically be downloaded during the install script. 

### Train RL (starting with a network initialized through supervised learning on CADRL decisions)

To start a GA3C training run (it should get approx -0.05-0.05 rolling reward to start):
```bash
./train.sh TrainPhase1
```

<!-- It should produce an output stream somewhat like this:
<img src="docs/_static/terminal_train_phase_1.gif" alt="Example output of terminal">
 -->

To load that checkpoint and continue phase 2 of training, update the `LOAD_FROM_WANDB_RUN_ID` path in `Config.py` and do:
```bash
./train.sh TrainPhase2
```

By default, the RL checkpoints will be stored in `RL_tmp` and I think files will get overwritten if you train multiple runs. Instead, I like using `wandb` as a way of recording experiments/saving network parameters. To enable this, set the `self.USE_WANDB` flag to be `True` in `Config.py`, then checkpoints will be stored in `RL/wandb/run-<datetime>-<id>`.

### To run experiments on AWS
Start a bunch (e.g., 5) of AWS instances -- I used `c5.2xlarge` because they have 8vCPUs and 16GB RAM (somewhat like my desktop?). Note: this is just an example, won't work out of box for you (has hard-coded paths)

Add the IP addresses into `ga3c_cadrl_aws.sh`.
```bash
./ga3c_cadrl_aws.sh panes
# C-a :setw synchronize-panes -- will let you enter the same command in each instance
```

Then you can follow the install & train instructions just like normal. When training, it will prompt you for a wandb login (can paste in the authorization code from app.wandb.ai/authorize).

### Observed Issues
If on OSX, when running the `./train.sh` script, you see:
```bash
objc[39391]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called.
objc[39391]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called. We cannot safely call it or ignore it in the fork() child process. Crashing instead. Set a breakpoint on objc_initializeAfterForkError to debug.
```
just add this ENV_VAR: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`.

### If you find this code useful, please consider citing:

```
@inproceedings{Everett18_IROS,
  address = {Madrid, Spain},
  author = {Everett, Michael and Chen, Yu Fan and How, Jonathan P.},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  date-modified = {2018-10-03 06:18:08 -0400},
  month = sep,
  title = {Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning},
  year = {2018},
  url = {https://arxiv.org/pdf/1805.01956.pdf},
  bdsk-url-1 = {https://arxiv.org/pdf/1805.01956.pdf}
}
```
```
setup(
    name='ga3c',
    version='1.0.0',
    description='Simulation environment for collision avoidance',
    # url='https://github.com/mit-acl/gym-collision-avoidance',
    author='Michael Everett, Yu Fan Chen, Jonathan P. How, MIT',  # Optional
    # keywords='robotics planning gym rl',  # Optional
    python_requires='>=3.0, <4',
    install_requires=[
        'tensorflow==1.15.2',
        'wandb',
        # 'Pillow',
        # 'PyOpenGL',
        # 'pyyaml',
        # 'matplotlib>=3.0.0',
        # 'shapely',
        # 'pytz',
        # 'imageio==2.4.1',
        # 'gym',
        # 'moviepy',
        # 'baselines',
    ],
)

```
train.sh
```bash
#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh
export GYM_CONFIG_CLASS=${1:-TrainPhase2}
export GYM_CONFIG_PATH=${2:-$DIR/ga3c/GA3C/Config.py}
# Train tf 
print_header "Running GA3C-CADRL gym-collision-avoidance training script (${GYM_CONFIG_CLASS})"
cd $DIR/ga3c/GA3C
# wandb off
python Run.py
```
utils.sh

```bash
#!/bin/bash
set -e

function print_header(){
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -    
echo $1
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Disable all tensorflow warnings/info (keep errors)
export TF_CPP_MIN_LOG_LEVEL=2

# Directory of this script
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR=${THIS_DIR}
source $BASE_DIR/venv/bin/activate
echo "Entered virtualenv."
export PYTHONPATH=${BASE_DIR}/venv/lib/python3.5/site-packages
```