from setuptools import setup

setup(
    name="baestorm",
    version="0.0.2",
    description="Bayesian Optimization Control Algorithms for Stormwater Control",
    author="Abhiram Mullapudi",
    author_email="abhiramm@umich.edu",
    url="https://github.com/kLabUM/baeopt",
    packages=["baestorm"],
    package_data={"baestorm": ["networks/*.inp"]},
    install_requires=["numpy", "matplotlib", "seaborn"],
)
