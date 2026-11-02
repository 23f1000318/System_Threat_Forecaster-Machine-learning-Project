---
title: System Threat Forecaster
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: space_app.py
python_version: 3.12
---

# System Threat Forecaster

Machine-learning project for predicting whether a Windows machine is likely to be infected by malware from antivirus telemetry and system configuration data.

This project was developed for the Kaggle **System Threat Forecaster** competition. It explores machine-level security, operating-system, hardware, geographic, and antivirus attributes, then trains a binary classifier to predict the competition target.

## Project overview

Each record represents a machine identified by `MachineID`. The target is a binary label indicating whether malware was detected on that machine. The task is therefore a supervised binary-classification problem.

The notebook covers the complete workflow:

- loading and inspecting the training and test data;
- exploratory data analysis of numerical and categorical features;
- analysis of missing values, duplicates, class balance, correlations, and feature distributions;
- preprocessing mixed data types with imputation, scaling, ordinal encoding, and one-hot encoding;
- extracting year, month, and day from `DateAS` and `DateOS`;
- removing duplicate rows and low-information or identifier-like features;
- clipping numerical outliers using the interquartile range (IQR);
- selecting the 200 most useful transformed features with a Random Forest-based `SelectFromModel` step;
- comparing Logistic Regression, Random Forest, and LightGBM classifiers;
- tuning LightGBM with `RandomizedSearchCV`;
- generating predictions in the required Kaggle submission format.

## Dataset

The competition data contains:

- `train.csv`: 100,000 rows and 76 columns, including `target`;
- `test.csv`: 10,000 rows and 75 feature columns;
- `sample_submission.csv`: example submission format.

The dataset is not included in this repository because it is provided under the Kaggle competition rules. Place the competition files in the appropriate Kaggle input directory before running the notebook.

### Feature groups

The available columns describe several aspects of each system:

- **Antivirus telemetry:** `ProductName`, `EngineVersion`, `AppVersion`, `SignatureVersion`, `RealTimeProtectionState`, `AntivirusConfigID`, and antivirus-product counts;
- **Security settings:** TPM, passive mode, firewall, system protection, secure boot, automatic sample submission, and related configuration flags;
- **Operating system:** OS version/build information, platform, edition, architecture, locale, installation type, update settings, and license state;
- **Hardware:** processor details, RAM, disk capacity/type, display properties, chassis, battery, firmware, and device capabilities;
- **Environment and geography:** country, city, region, OEM, device family, and power-platform role;
- **Dates:** `DateAS` (malware-signature date) and `DateOS` (OS-version/update timestamp);
- **Target:** binary malware-detection label available only in the training data.

## Exploratory findings

The notebook reports the following observations:

- the target classes are approximately balanced, with labels `0` and `1`;
- the training data contains duplicate records and 165 repeated rows were removed during preprocessing;
- missing values occur across multiple columns, with `SMode` having the largest reported number of missing values;
- some columns are constant or nearly non-informative, including `IsBetaUser`, `AutoSampleSubmissionEnabled`, and `IsFlightsDisabled`;
- several numerical and categorical features show associations with the target, although correlation alone is not used as the final feature-selection method;
- one-hot encoding expands the feature space substantially, from the original mixed schema to 3,495 transformed features before selection.

## Preprocessing and feature engineering

The preprocessing logic in the notebook performs the following operations:

1. Removes duplicate training rows.
2. Drops columns with more than 30% missing values.
3. Removes `MachineID` and selected low-information or redundant columns.
4. Converts `DateAS` and `DateOS` into year, month, and day components.
5. Clips numerical outliers using IQR-based lower and upper bounds on the training data.
6. Imputes numerical values with mean or median, depending on skewness.
7. Standardizes numerical features.
8. Imputes categorical values with the most frequent category.
9. Applies ordinal encoding to low-cardinality categorical columns and one-hot encoding to the remaining categorical columns.
10. Selects 200 transformed features using a Random Forest estimator.

The fitted preprocessing and feature-selection objects are reused for the test data so that train and test features remain aligned.

## Models and results

The notebook evaluates models using `accuracy_score`, matching the competition metric.

| Model | Validation accuracy | Notes |
| --- | ---: | --- |
| Dummy classifier | 0.507 | Baseline close to random/class-frequency performance |
| Logistic Regression | 0.605 | Training and validation scores were relatively close |
| Random Forest | 0.615 | Strong training score but noticeable overfitting |
| LightGBM | 0.625 | Best untuned validation result among the initial models |
| Tuned LightGBM | 0.623 | Higher training score but no validation improvement |

The final submission model is a LightGBM classifier trained on the full preprocessed training set. The notebook reports a submission accuracy of approximately **0.63110**.

The selected LightGBM configuration is:

```text
num_leaves=400
learning_rate=0.01
num_iterations=900
max_bin=300
boosting='gbdt'
random_state=42
```

## Repository contents

```text
.
├── 23f1000318-notebook-t12025.ipynb  # EDA, preprocessing, modeling, and submission workflow
├── train.py                           # Trains and serializes the deployment model
├── app.py                             # FastAPI prediction service
├── space_app.py                       # Gradio app for Hugging Face Spaces
├── streamlit_app.py                   # Streamlit Community Cloud app
├── requirements.txt                   # Runtime and training dependencies
├── Dockerfile                         # Container configuration
├── model/                             # Generated model artifact directory
└── README.md                          # Project documentation
```

## Running the project

The notebook was created for the Kaggle Python environment and expects the competition files under:

```text
/kaggle/input/System-Threat-Forecaster/
```

To reproduce the workflow:

1. Download the competition data from Kaggle.
2. Open `23f1000318-notebook-t12025.ipynb` in Kaggle Notebooks or a compatible Jupyter environment.
3. Ensure `train.csv`, `test.csv`, and `sample_submission.csv` are available at the expected input path.
4. Run the notebook from beginning to end.
5. The final cells train LightGBM on the complete training data and write `submission.csv` with `id` and `target` columns.

The main Python dependencies are NumPy, pandas, scikit-learn, Matplotlib, Seaborn, and LightGBM.

## Submission format

The generated file follows the competition format:

```csv
id,target
0,0
1,1
2,1
```

## Deploying the model as an API

The repository now includes a FastAPI service for online predictions. The notebook remains unchanged; deployment uses the same preprocessing and LightGBM approach through `train.py` and `app.py`.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create the model artifact

Place the Kaggle training file at `data/train.csv`, then run:

```bash
python train.py --train-csv data/train.csv --output model/model.joblib
```

The generated model bundle is intentionally ignored by Git because it can be large and the competition data is not part of this repository.

### 3. Start the API locally

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Check the service:

```bash
curl http://localhost:8000/health
```

Send a prediction by posting one machine's telemetry as a JSON object to `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ProductName":"Windows Defender","EngineVersion":"1.1.15200.1","HasTpm":1,"NumAntivirusProductsInstalled":1}'
```

The API accepts partial records; missing features are passed through the saved preprocessing pipeline as missing values. A response contains the predicted class and the estimated probability of malware detection.

### Docker

After generating `model/model.joblib`, build and run the service:

```bash
docker build -t system-threat-forecaster .
docker run --rm -p 8000:8000 system-threat-forecaster
```

## Hugging Face Spaces deployment

This repository is configured as a Gradio Space through the YAML metadata at the top of this README. Hugging Face will use `space_app.py` as the entry point and install packages from `requirements.txt`.

1. Create a new Space at [Hugging Face Spaces](https://huggingface.co/new-space).
2. Choose **Gradio** as the SDK and Python 3.11 as the runtime.
3. Choose **Private** visibility if the model artifact or competition data should not be public.
4. Upload the repository files, including `model/model.joblib`.
5. Wait for the Space to build, then open the generated app URL.

The Space supports both one-record JSON predictions and batch CSV predictions. The raw `train.csv` is not required at runtime; only the generated `model/model.joblib` artifact is needed. Do not upload Kaggle credentials or raw competition data to a public Space.

You can upload the files from the command line after creating the Space and authenticating locally with `hf auth login`:

```bash
HF_SPACE=your-huggingface-username/system-threat-forecaster

hf upload "$HF_SPACE" README.md --repo-type=space
hf upload "$HF_SPACE" app.py --repo-type=space
hf upload "$HF_SPACE" train.py --repo-type=space
hf upload "$HF_SPACE" space_app.py --repo-type=space
hf upload "$HF_SPACE" requirements.txt --repo-type=space
hf upload "$HF_SPACE" model/model.joblib model/model.joblib --repo-type=space
```

The model file is approximately 210 MB, so the upload may take some time. Hugging Face's `hf upload` command supports large files and can resume interrupted uploads.

## Free deployment with Streamlit Community Cloud

Streamlit Community Cloud is the recommended free deployment option for this project. It connects to GitHub repositories and handles the application environment and deployment automatically. See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).

Before deploying, upload `model/model.joblib` to GitHub with Git LFS because the artifact is larger than GitHub's normal 100 MB file limit:

```bash
brew install git-lfs
git lfs install
git lfs track "model/model.joblib"
git add .gitattributes streamlit_app.py requirements.txt train.py
git add -f model/model.joblib
git commit -m "Add Streamlit deployment"
git push origin main
```

Then:

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
2. Select **Create app**.
3. Choose this GitHub repository and branch.
4. Set the main file to `streamlit_app.py`.
5. Click **Deploy**.

The app provides JSON predictions for one machine and batch predictions from a CSV upload. Streamlit Community Cloud is free, although free applications may sleep when inactive and are intended for demos and personal projects rather than production workloads.

## Notes and possible improvements

This repository documents the original notebook workflow and does not alter the notebook itself. Potential future improvements include stratified cross-validation, leakage-safe preprocessing inside a complete modeling pipeline, more systematic hyperparameter optimization, class-probability calibration, and comparison with CatBoost or other models that handle high-cardinality categorical features directly.

## Competition reference

The project is based on Kaggle's **System Threat Forecaster** competition, where the objective is to predict whether a machine will be infected by malware using antivirus and system telemetry.
