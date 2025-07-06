\# HousingRegression 



A machine learning workflow to predict housing prices using classical regression models and automate the pipeline with GitHub Actions CI.



---



\##  Assignment Objective



The goal is to:

\- Implement 3 classical regression models

\- Perform performance comparison using MSE and R²

\- Apply hyperparameter tuning (minimum 3 hyperparameters per model)

\- Follow a modular ML Ops workflow using Git and GitHub Actions



---



\##  Repository Structure



HousingRegression/

├── .github/

│ └── workflows/

│ └── ci.yml # GitHub Actions for CI pipeline

├── utils.py # Data loading and split utilities

├── regression.py # 3 regression models (reg branch)

├── hyperparameter\_tuning.py # GridSearchCV tuning (hyper branch)

├── requirements.txt # Python package requirements

└── README.md # This file





---



\##  Models Used



| Model               | Branch       |

|--------------------|--------------|

| Linear Regression  | `reg`, `hyper` |

| Decision Tree      | `reg`, `hyper` |

| Random Forest      | `reg`, `hyper` |



Each model is evaluated using:

\- Mean Squared Error (MSE)

\- R² Score



---



\##  Hyperparameter Tuning



Done in the `hyper` branch using `GridSearchCV`:

\- 3+ hyperparameters per model

\- 5-fold cross-validation

\- Best model selection based on performance



---



\##  CI/CD with GitHub Actions



Automated testing is set up in `.github/workflows/ci.yml` and runs on every push to:



\- `main`

\- `reg`

\- `hyper`



Steps:

1\. Checkout code

2\. Install dependencies from `requirements.txt`

3\. Run regression script (main branch runs `regression.py`)



---



\## Dataset



The Boston Housing dataset is manually downloaded from:



http://lib.stat.cmu.edu/datasets/boston



Dataset is parsed in `utils.py` using:

```python

data\_url = "http://lib.stat.cmu.edu/datasets/boston"





How to Run Locally

Clone the repository



Create a conda environment:



conda create -n housing\_env python=3.10 -y

conda activate housing\_env

pip install -r requirements.txt





Run regression:



python regression.py



Run hyperparameter tuning:



python hyperparameter\_tuning.py

