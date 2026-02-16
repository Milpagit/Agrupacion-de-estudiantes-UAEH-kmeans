# Student Grouping and Approval Prediction for UAEH

A Python-based project for student grouping using K-means and predicting academic approval.

## Overview

This repository contains scripts and pre-trained models designed to facilitate the analysis and grouping of students at UAEH (Universidad Autónoma del Estado de Hidalgo). It includes modules for data preprocessing, K-means clustering operations, and a regression model to predict student approval, along with its associated preprocessor. The core aim is to provide tools for educational data mining to understand student demographics and performance better.

## Tech Stack

| Technology | Category    | Notes                                  |
| :--------- | :---------- | :------------------------------------- |
| Python     | Language    | Primary development language.          |
| `.pkl`     | Data Format | Used for serializing Python objects, likely machine learning models and preprocessors. |
| `.csv`     | Data Format | Input data format.                     |

## Features

*   **Data Preprocessing**: Dedicated script for preparing raw student data for analysis.
*   **Student Clustering**: Implements K-means clustering algorithms to group students based on defined criteria.
*   **Approval Prediction Model**: Includes a pre-trained regression model for predicting student approval outcomes.
*   **Reusable Preprocessors**: Provides a preprocessor specifically for the approval prediction model.
*   **Data Input**: Utilizes CSV files for student data.

## Prerequisites

To run this project, you will need:

*   **Python**: Version 3.x (specific version not specified in repository data).

_Note: While the use of `.pkl` files strongly implies libraries like `scikit-learn`, these are not explicitly mentioned in the provided repository data. Therefore, specific Python library prerequisites cannot be listed here._

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Milpagit/Agrupacion-de-estudiantes-UAEH-kmeans.git
    cd Agrupacion-de-estudiantes-UAEH-kmeans
    ```

2.  **Install dependencies**:
    As no `requirements.txt` or similar dependency file is provided, you may need to manually install any necessary Python libraries based on script execution errors. A common command for installing Python packages is `pip`.

    ```bash
    # Example: If a library like pandas or scikit-learn is missing, you might install it like this:
    # pip install pandas scikit-learn
    ```

## Usage

This project involves running Python scripts for data preprocessing, clustering, and potentially using the pre-trained models.

1.  **Prepare your data**:
    Ensure your student data is available in a CSV file, similar to `Datos.csv`.

2.  **Run data preprocessing**:
    Execute the preprocessing script to clean and prepare your data.
    ```bash
    python Preprosesamiento_Datos.py
    ```
    _Note: Specific input arguments or output details for this script are not available from the provided data._

3.  **Perform student clustering**:
    Execute the K-means clustering script.
    ```bash
    python Op1.py
    # or
    python Op1_unificado.py
    ```
    _Note: Details on how to specify input data, number of clusters, or output format are not available._

4.  **Utilize the prediction model**:
    The `modelo_regresion_aprobacion.pkl` and `preprocessor_regresion.pkl` files suggest a regression model for approval prediction. You would typically load these files within a Python script to make predictions on new data.
    ```python
    import pickle

    # Example of loading a model (specific usage depends on your script)
    with open('modelo_regresion_aprobacion.pkl', 'rb') as f:
        approval_model = pickle.load(f)

    with open('preprocessor_regresion.pkl', 'rb') as f:
        reg_preprocessor = pickle.load(f)

    # You would then use reg_preprocessor to transform new data
    # and approval_model to make predictions.
    ```
    _Note: There is no specific script provided to demonstrate the direct usage of the prediction model files._

## Scripts

No specific build or utility scripts (e.g., in a `package.json` or `Makefile`) are provided in the repository data. All operations are expected to be executed directly via Python commands as shown in the Usage section.

## Folder structure

| Path                            | Type   | Short purpose                                     |
| :------------------------------ | :----- | :------------------------------------------------ |
| `Datos.csv`                     | file   | Sample or input dataset for student information.  |
| `Op1.py`                        | file   | Python script for a specific operation, likely clustering. |
| `Op1_unificado.py`              | file   | Unified Python script for a specific operation, likely clustering. |
| `Preprosesamiento_Datos.py`     | file   | Python script for data preprocessing tasks.       |
| `modelo_regresion_aprobacion.pkl` | file   | Pre-trained regression model for student approval prediction. |
| `preprocessor_regresion.pkl`    | file   | Preprocessor object for the regression model, likely for data transformation. |

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these general guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure they adhere to existing code style.
4.  Write clear, concise commit messages.
5.  Submit a pull request with a detailed description of your changes.

## License

The license for this project is not specified in the provided repository data.
