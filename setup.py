
#%%
from setuptools import setup, find_packages
from scoring import __version__
#%%
setup(
    name="scoring",
    version=__version__,
    description="""Scoring library""",
    url="https://git.homecredit.net/risk/python-scoring-workflow",
    author="Home Credit HQ Risk Research & Development",
    author_email="marek.mukensnabl@homecredit.eu",
    license="Apache 2.0",
    packages=find_packages(),

    package_data={'scoring': [r"fefe_core\front_end\assets\*"]}
)
