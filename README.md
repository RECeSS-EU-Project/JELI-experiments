# Code for "Joint Embedding–Classifier Learning for Interpretable Collaborative Filtering"

**Licence**: MIT

## 1. System requirements

**OS**: Linux Debian **Python version**: 3.8.*

The dependencies, along with package versions, are listed in the file named *requirements.txt*.

## 2. Installation guide (estimated install time: ~15min)

It is strongly advised to install the [Conda](https://docs.anaconda.com/free/miniconda/miniconda-install/) tool to create a virtual environment, and [Pip](https://pip.pypa.io/en/stable/installation/) for installing dependencies:

```bash
conda create --name jeli_envir python=3.8 -y
conda activate jeli_envir
```

Once the virtual environment is created and activated:

```bash
python3 -m pip install pip==24.0 benchscofi==2.0.0 --no-dependencies --no-cache-dir 
python3 -m pip install -r requirements.txt #--no-cache-dir 
python3 -m pip install torch
```

## 3. Reproduction of experiments

- To reproduce Figure 1 and Table 1, execute *tests_interpretable.py*.

- To reproduce Figures 2-3 and Tables 2-3, execute *tests_sparsity.py*.

- To reproduce Figure 4 and Table 4, execute *tests_drug_repurposing.py*.

- To reproduce Figure 5, Table 5 and Tables 7-8 (appendix), execute *tests_add_prior.py* (for the None and STRING priors) and *tests_add_prior2.py* (for the remainder of priors).

- To reproduce Figures 6 and 7 (appendix), execute *tests_parameter.py* (for true dimension parameter d=2) and *tests_parameter2.py* (for true dimension parameter d=5).

At the end of all runs, execute *plots_tables.py* to generate tables and figures.

## 4. Citation

```bash
@article{reda2025joint,
  title={Joint embedding--classifier learning for interpretable collaborative filtering},
  author={R{\'e}da, Cl{\'e}mence and Vie, Jill-J{\^e}nn and Wolkenhauer, Olaf},
  journal={BMC bioinformatics},
  volume={26},
  number={1},
  pages={26},
  year={2025},
  publisher={Springer}
}

```
