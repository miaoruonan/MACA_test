# gym-collision-avoidance (ENV: MACA)

<img src="asset/combo.gif" alt="Agents spelling ``CADRL''">

This is the code associated with the following publications:

**Journal Version:** M. Everett, Y. Chen, and J. P. How, "Collision Avoidance in Pedestrian-Rich Environments with Deep Reinforcement Learning", in review, [Link to Paper](https://arxiv.org/abs/1910.11689)

**Conference Version:** M. Everett, Y. Chen, and J. P. How, "Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018. [Link to Paper](https://arxiv.org/abs/1805.01956), [Link to Video](https://www.youtube.com/watch?v=XHoXkWLhwYQ)

This repo also contains the trained policy for the SA-CADRL paper (referred to as CADRL here) from the proceeding paper: Y. Chen, M. Everett, M. Liu, and J. P. How. “Socially Aware Motion Planning with Deep Reinforcement Learning.” IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Vancouver, BC, Canada, Sept. 2017. [Link to Paper](https://arxiv.org/abs/1703.08862)

If you're looking to train our GA3C-CADRL policy, please see [this repo](https://github.com/mit-acl/rl_collision_avoidance) instead.

---

### About the Code

Please see [the documentation](https://gym-collision-avoidance.readthedocs.io/en/latest/)!

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
    name='gym_collision_avoidance',
    version='1.0.0',
    description='Simulation environment for collision avoidance',
    url='https://github.com/mit-acl/gym-collision-avoidance',
    author='Michael Everett, Yu Fan Chen, Jonathan P. How, MIT',  # Optional
    keywords='robotics planning gym rl',  # Optional
    python_requires='>=3.0, <4',
    install_requires=[
        'tensorflow==1.15.2',
        'Pillow',
        'PyOpenGL',
        'pyyaml',
        'matplotlib>=3.0.0',
        'shapely',
        'pytz',
        'imageio==2.4.1',
        'gym',
        'moviepy',
        'pandas',
    ],
)
```
