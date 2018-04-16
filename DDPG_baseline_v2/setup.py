from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='baselines_cecolas',
      packages=[],
            # package for package in find_packages()
            #     if package.startswith('baselines')],
      install_requires=[],
      #     'gym[mujoco,atari,classic_control]',
      #     'scipy',
      #     'tqdm',
      #     'joblib',
      #     'zmq',
      #     'dill',
      #     'tensorflow >= 1.0.0',
      #     'azure==1.0.3',
      #     'progressbar2',
      #     'mpi4py',
      # ],
      description="DDPG implementation modified from OpenAI baselines (https://github.com/openai/baselines)",
      author="OpenAI",
      url='https://gitlab.inria.fr/cecolas/DDPG_baseline_v2',
      author_email="cedric.colas@inria.fr",
      version="0.0.1")
