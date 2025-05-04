# MNLP 2025: Cultural Classifier (Non-LM Approach)

This repository contains the source code, data, and documentation for our rule-based and hybrid (rule + ML fallback) cultural classifier developed for the MNLP 2025 Homework 1 project.

## Project Overview

The goal of the project is to classify Wikidata entities into one of the following cultural categories:

* **Cultural Agnostic**
* **Cultural Representative**
* **Cultural Exclusive**

Our approach is based entirely on structured metadata from Wikidata and does not rely on any pretrained language models. We built a deterministic rule-based classifier using manually curated QID lists ("golden QIDs") and extended this with a fallback ML classifier to handle ambiguous or uncovered cases.

---

## Directory Structure

```plaintext
NONLM_APPROACH/
│
├── analyze_test_results.py           # Diagnostic script for test predictions
├── analyze_validation_results.py     # Diagnostic script for validation predictions
├── apply_fallback_test.py            # Applies ML fallback on test set
├── apply_fallback_validation.py      # Applies ML fallback on validation set
├── run_RBC_test.py                   # Runs rule-based classifier on test set
├── run_RBC_validation.py             # Runs rule-based classifier on validation set
├── README.md                         # Project documentation
├── requirements.txt                  # Required packages
│
├── data/                             # Raw and enriched datasets
│   ├── golden_culture_qids.csv       # Curated QIDs for part_of_culture
│   ├── golden_heritage_qids.csv      # Curated QIDs for heritage_status
│   ├── golden_instance_qids.csv      # Curated QIDs for instance_of
│   ├── train.csv                     # Original training dataset
│   ├── train_enriched.csv            # Training dataset after Wikidata enrichment
│   ├── train_enriched_labeled.csv    # Enriched training set with labels
│   ├── validation_enriched.csv       # Enriched validation set (no labels)
│   ├── validation_labeled.csv        # Raw labeled validation data
│   ├── validation_enriched_with_labels.csv
│   ├── validation_raw.csv            # Raw validation (for merging or diagnostics)
│   ├── test_enriched.csv             # Enriched test dataset
│   ├── test_unlabeled.csv            # Raw test set
│
├── models/
│   └── tfidf_fallback_model.pkl      # Trained logistic regression model for fallback
│
├── notebooks/
│   └── teXt_Men_HW1_eval.ipynb       # Evaluation and visualization notebook
│
├── results/
│   ├── test_rules.csv                # Rule-based predictions (test)
│   ├── hybrid_test_final_predicts.csv # Final hybrid predictions (test)
│   ├── validation_rules.csv          # Rule-based predictions (validation)
│   ├── validation_fallback.csv       # Hybrid fallback predictions (validation)
│
├── scripts/
│   ├── constants.py                  # Wikidata property mappings for cultural enrichment
│   └── wikidata_enrichment.py        # Script for metadata enrichment via Wikidata
```

---

## Methodology

### Rule-Based Classification (RBC)

Our rule-based classifier assigns a cultural label based on three key Wikidata properties:

* `heritage_status`
* `part_of_culture`
* `instance_of`

We follow a strict priority logic:

1. If any QID in `heritage_status` matches a golden heritage QID → **Cultural Exclusive**
2. Else if any QID in `part_of_culture` matches a golden culture QID → **Cultural Representative**
3. Else if any QID in `instance_of` matches a golden instance QID → **Based on match**
4. Else → **Cultural Agnostic**

The golden QID lists were created through a multi-stage process:

* Initial curation from correct samples
* Diagnostics on misclassifications
* Frequency-based filtering of key `instance_of` values
* Manual inspection and exclusion of overly broad or irrelevant QIDs

### ML Fallback Classifier

The fallback model is a logistic regression trained on TF-IDF features derived from enriched `name + description` fields. If the rule-based classifier assigns a label of **Cultural Agnostic**, and sufficient textual metadata is present, the ML model reclassifies the item.

This fallback is only invoked selectively, preserving the determinism of rule-based matches while allowing soft corrections elsewhere.

---



### Setup

```bash
pip install -r requirements.txt
```

---

## Model Training

The logistic regression fallback was trained using `train_enriched_labeled.csv`. The model file (`tfidf_fallback_model.pkl`) is available under `/models`. The training script is not included here for brevity but follows a simple TF-IDF + LogisticRegression pipeline with `max_iter=1000`.



---

## Contact

For questions, reach out to:

* Joshua Edwin — [vijayanrajendraedwin.1995743@studenti.uniroma1.it](mailto:vijayanrajendraedwin.1995743@studenti.uniroma1.it)
