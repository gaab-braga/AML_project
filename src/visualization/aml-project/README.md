# AML Project

## Overview
The AML (Anti-Money Laundering) project aims to develop and optimize machine learning models for detecting suspicious transactions. The project includes data processing, model training, evaluation, and visualization components.

## Project Structure
```
aml-project/
├── data/
│   └── processed/                # Contains processed data files for model training and evaluation
├── notebooks/
│   └── 03_Modelagem_e_Avaliacao.ipynb  # Main analysis and modeling code
├── src/
│   ├── __init__.py               # Marks the src directory as a Python package
│   ├── config.py                  # Configuration settings for the project
│   ├── data_processing.py          # Functions for data loading, preprocessing, and feature engineering
│   ├── model_training.py           # Functions for training machine learning models
│   └── visualization.py            # Functions for visualizing results
├── requirements.txt                # Lists required Python packages
└── README.md                       # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd aml-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
- Use the Jupyter notebook located in `notebooks/03_Modelagem_e_Avaliacao.ipynb` for the main analysis and modeling workflow.
- Modify the configuration settings in `src/config.py` as needed for your environment.
- Utilize the functions in `src/data_processing.py` for data preprocessing and feature engineering.
- Train models using the functions in `src/model_training.py`.
- Visualize results with the functions in `src/visualization.py`.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License. See the LICENSE file for details.