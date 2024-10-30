# Specific enrichments and independent validations

**Diseases:** The following list is considered (file *diseases.csv*):

- Melanoma (MedGen Concept ID: C0025202)

**Licence**: MIT

## Installation (collaborative filtering with JELI)

**OS**: Linux Debian **Python version**: 3.8.*

The dependencies, along with package versions, are listed in the file named *requirements.txt*.

```bash
conda create --name jeli_envir python=3.8 -y
conda activate jeli_envir
python3 -m pip install -r requirements.txt --no-cache-dir 
conda clean -a -y
```
## 1. Enrichment

**Objective**: Connect the predicted drug-disease matchings to biologically meaningful information.

**Methodology**: Generate importance score per gene for each considered disease with the [JELI](https://github.com/recess-eu-project/jeli) algorithm run on the [TRANSCRIPT](https://github.com/recess-eu-project/drug-repurposing-datasets) data set. Run a Gene Set Enrichment Analyses (GSEA) to perturbed Gene Onthology and KEGG pathways with WebGestalt (number of permutations 10,000; false discovery rate < 20%).

**Validation**: The enriched pathways correspond to known mechanisms of the disease and of the drug(s) as provided by literature.

## 2. Independent validation

**Objective**: Test whether the predictions from the collaborative filtering approach align with the prediction made by protein docking.

**Methodology**: Get drug-disease matchings with the highest scores for each considered disease from the [JELI](https://github.com/recess-eu-project/jeli) algorithm run on the [TRANSCRIPT](https://github.com/recess-eu-project/drug-repurposing-datasets) data set. Look for clues in the recent literature.

**Validation**: Prior drugs match (the rank of) recommended drugs for each considered disease. 
