from setuptools import setup, find_packages

setup(
    name="fdi_rl_panda",
    version="2.0.0",
    description="Reinforcement Learning for Transformer Control with FDI Attack Detection",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "pandapower",
        "gymnasium",
        "stable-baselines3[extra]",
        "torch",
        "tensorboard",
    ],
    python_requires=">=3.8",
)
