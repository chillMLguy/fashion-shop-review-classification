# Fashion Shop Review Classification

Repository: chillMLguy / fashion-shop-review-classification  
Notebooks:
- [main.ipynb](https://github.com/chillMLguy/fashion-shop-review-classification/blob/main/main.ipynb)
- [EDA.ipynb](https://github.com/chillMLguy/fashion-shop-review-classification/blob/main/EDA.ipynb)

Overview
--------
This project trains a model to classify clothing product reviews as recommended (1) or not recommended (0). It contains exploratory data analysis (EDA) and a modelling notebook that builds pipelines for numerical, categorical, and text features and fits machine learning models.

Data
----
- Expected CSV: `reviews.csv` (must be placed in the repository root or the working directory when running the notebooks).
- The dataset used in the notebooks comes from an Udacity course and includes columns such as:
  - `Clothing ID` (dropped during preprocessing)
  - `Age`
  - `Title`
  - `Review Text`
  - `Positive Feedback Count`
  - `Division Name`, `Department Name`, `Class Name`
  - `Recommended IND` (target)

Repository files
----------------
- `EDA.ipynb` — Exploratory Data Analysis:
  - Loads `reviews.csv`
  - Shows sample rows, shape, info and missing values
  - Splits features into numerical, categorical, and text
  - Plots distributions for the features (histograms and countplots)
  - Helps understand class balance and feature distributions

- `main.ipynb` — Modelling pipeline and experiments:
  - Imports libraries (NumPy, pandas, scikit-learn, spaCy, HuggingFace transformers, PyTorch, matplotlib, seaborn)
  - Loads `reviews.csv`, drops `Clothing ID`, splits into X/y and train/test
  - Builds three pipelines:
    - Numerical pipeline using `StandardScaler`
    - Categorical pipeline using `OrdinalEncoder` and `OneHotEncoder`
    - (Text pipeline is referenced and transformers/spaCy are imported — this notebook demonstrates integrating text features into a ColumnTransformer / pipeline)
  - Combines pipelines via `ColumnTransformer` or `FeatureUnion`
  - Trains a classifier (RandomForestClassifier is used as an example) and performs GridSearchCV
  - Evaluates with accuracy, confusion matrix and classification report

Requirements
------------
Recommended Python: 3.8+ (not strictly required; notebooks used Python 3.10 in metadata)

Minimum Python packages (example installation):
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- spacy
- transformers
- torch (PyTorch)
- jupyter (or JupyterLab / Colab)

Install via pip:
```
pip install numpy pandas matplotlib seaborn scikit-learn spacy transformers torch jupyter
```
of requirements.txt:
```
pip install requirements
```
If you use spaCy models (noted in `main.ipynb`):
```
python -m spacy download en_core_web_sm
```

How to run
----------
1. Clone the repository:
   ```
   git clone https://github.com/chillMLguy/fashion-shop-review-classification.git
   cd fashion-shop-review-classification
   ```
2. Ensure `reviews.csv` is in the repository root (or update the path in the notebooks).
3. Install dependencies (see Requirements).
4. Open and run notebooks:
   - In a local environment:
     ```
     jupyter notebook
     ```
     then open `EDA.ipynb` and `main.ipynb`.
   - Or open notebooks in Google Colab (upload `reviews.csv` to Colab or point to a cloud-hosted file).

Notebook summaries
------------------
- EDA.ipynb
  - Quick look at data (head, shape, dtypes)
  - Checks for missing values (none in the checked columns)
  - Visualizes distributions for numeric and categorical features
  - Produces insight into the dataset and class balance

- main.ipynb
  - Prepares preprocessing pipelines:
    - Numeric: Standard scaling
    - Categorical: Ordinal encoding + OneHot encoding (handling unknowns)
    - Text: notebook imports spaCy and Hugging Face transformers (tokenizers + models) — text processing is prepared to be integrated into the pipeline
  - Splits the dataset into train/test (10% test)
  - Demonstrates training a Random Forest and doing GridSearch for hyperparameters
  - Evaluates model using accuracy, confusion matrix and classification report

Reproducing results and tips
----------------------------
- Make sure `reviews.csv` matches the format used in the notebooks. Column names must match exactly.
- If running transformer-based text embeddings, ensure you have enough memory and compute and the correct PyTorch / transformers versions.
- For reproducibility, set random seeds (the notebook uses `random_state=42` for train/test split).

Potential improvements / next steps
----------------------------------
- Add more robust text preprocessing (lowercasing, stopword removal, lemmatization — possibly with spaCy).
- Use transformer embeddings (e.g., sentence-transformers / Hugging Face models) and freeze or fine-tune depending on compute.
- Evaluate and mitigate class imbalance if present (resampling, class weights).
- Compare more models (Logistic Regression, XGBoost, LightGBM, neural networks).
- Add unit tests and a small sample of the dataset for quick CI checks.

Contact / author
----------------
GitHub: [chillMLguy](https://github.com/chillMLguy)

License
-------
Distributed under the MIT License.

Acknowledgements
----------------
Dataset referenced from an Udacity course used for learning classification of clothing reviews.

```
