# Python Scoring library and workflows


On this page, you can find always the most recent version of PSW (Python Scoring Workflow) and its underlying Scoring library.

## Contents
- This README
- [Feature Engineering (FEFE) readme](info/FEFE_README.md)
- [Installation Troubleshooting](info/install_troubleshooting.md)
- [Frequently Asked Questions](info/faq.md)
- [PSW changelog](CHANGELOG.md)
- [Contributing to PSW](CONTRIBUTING.md)
- [License](LICENSE.md)

## Getting Started



1) [Install Anaconda from here.](https://www.anaconda.com/download/)  
 This will download full Anaconda distribution with many other packages, including jupyter notebooks.  
__IMPORTANT__: During installation, click on '*Add Anaconda to path*', even though this option is discouraged!

2) Run `install.bat` (in this repository) and follow the instructions.

<!-- We recommend Python 3.6, 64-bit version (so it can utilize enough RAM), in Anaconda distribution (which is gives you most of the common libraries and ability to install more from Anaconda prompt). As the environment for our work, we use Jupyter Notebooks. Jupyter is a part of Anaconda installation. [You can get Anaconda here.](https://www.anaconda.com/download/) -->


## Which workflows are currently available?

Apart from *Python Scoring Workflow*, library was extended with, following, available in [`/workflow`](/workflow):

- *General Model workflow*
- *Target Analysis*
- *Gradient boosting workflow*
- *Data Download*
- *Data Preparation 1*
- *Data Preparation 2*
- *Feature Engineering FrontEnd*
- *China FeatureTools Demo*
- *Collection data preparation*
- *Collection model evaluation*

## How to start the workflow?
0) Install git client:
    - [git commandline](https://git-scm.com/download/win)  
    For easier usage you can also install a graphical interface
        - [tortoise git](https://tortoisegit.org/)  
        or/and
        - [fork](https://git-fork.com/)
1) Clone this repository to a local folder
    - [using commandline](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html#clone-a-repository)
    - [using tortoise git](https://tortoisegit.org/docs/tortoisegit/tgit-dug-clone.html)
    - [using fork](https://www.presslabs.com/docs/development/git/fork-windows/#clone-your-repository)
2) Start Jupyter notebook using *anaconda prompt* and writing `jupyter notebook`
3) New tab will appear in your browser
4) Open some of the `.ipynb` workflows from `\workflow` folder

**IMPORTANT:** Do not close the command prompt window called Jupyter Notebook, this is where your Python Kernel is running.  
In case of problems, write an [issue here](https://git.homecredit.net/risk/python-scoring-workflow/issues).

<!-- Just place the .ipynb file, scoring folder (unzipped from the .zip archive) and .csv files somewhere in your Documents folder, then start Jupyter Notebook. A new tab in your web browser should appear where you can open files in your Documents folder (more generally in folder which is mapped for Jupyter to “see into”, which is the Documents folder by default). Open the .ipynb file and you can start working. Do not close the command prompt window called Jupyter Notebook, because there is your Python kernel running. -->






## Contributors

* [**Pavel Sůva**](mailto:pavel.suva@homecredit.eu) (HCI Research & Development)
* **Sergey Gerasimov** (HCRU Scoring & Big Data)
* **Valentina Kalenichenko** (HCRU Scoring & Big Data)
* **Marek Teller**
* **Martin Kotek** (HCCN Risk Decision & Big Data)
* **Vítězslav Klepetko** (HCPH Big Data & Scoring)
* **Jan Zeller**
* [**Marek Mukenšnabl**](mailto:marek.mukensnabl@homecredit.eu) (HCI Research & Development)
* **Anatoliy Glushenko** (HCRU Scoring & Big Data)
* **Kirill Odintsov** (HCID Risk)
* [**Jan Hynek**](mailto:jan.hynek@homecredit.eu) (HCI Research & Development)
* **Elena Kuchina** (HCI Research & Development)
* **Dmitry Gorev** (HCRU Scoring & Big Data)
* **Hynek Hilbert** (HCRU)
* **Lubor Pacák** (HCI Research & Development)
* [**Naďa Horká**](mailto:nada.horka@homecredit.eu) (HCI Research & Development)
* **Kamil Yazigee** (HCI Research & Development)

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details
