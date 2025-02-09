# Neural Trade

## Project Overview

The `Neural Trade` project is designed for financial market analysis, utilizing machine learning models, specifically Multi-Layer Perceptron (MLP) models, to predict market movements. The repository includes data processing, model training, evaluation, and backtesting capabilities. It utilizes various indicators and feature engineering techniques, such as log transformations, EMA, and percentage changes, to prepare data for model training.

The project leverages deep learning frameworks, such as TensorFlow/Keras, and is tailored to cryptocurrency market data (e.g., Bitcoin price predictions).

## Project Structure

neural_trade/ 
│ ├── data/ # Folder containing input data 
│ └── futures_um_monthly_klines_BTCUSDT_3m_0_7.csv # Example of the data file 
│ ├── models/ # Folder containing model definitions 
│ └── MLP.py # MLP model definition for regression or classification
│ ├── result/ # Folder to store results and output files
│ └── saved_model/ # Folder for saving trained models
│ └── mlp.keras # Example of a saved model
│ ├── utils/ # Utility scripts for data processing, indicators, etc.
│ ├── data_processing.py # Functions for data loading and transformations
│ ├── indicator.py # Indicator calculations like EMA and up/down volume
│ ├── analyse_model.py # Model analysis and summary
│ ├── backtest_sim.py # Backtesting simulation
│ └── train_and_save_model_main.py # Main script for model training
│ ├── .gitignore # Git ignore file
├── LICENSE # Project license
├── requirements.txt # Python dependencies
├── strategy/ # Folder for strategies
└── train_and_save_model_main.py # Main script for training and saving the model

## Installation

1. Clone or download this repository to your local machine.
2. Set up a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
3. Install dependencies from requirements.txt:   
pip install -r requirements.txt

Usage
1. Data Loading and Preprocessing
To load data and apply transformations, run the train_and_save_model_main.py script. This script processes input data and prepares it for model training.

python utils/train_and_save_model_main.py
This script will process data, apply feature engineering techniques, such as calculating percentage changes, EMA, and log transformations, and then prepare the data for model training.

2. Model Training and Evaluation
You can train the MLP model using the following command:

python utils/train_and_save_model_main.py
This script will train the MLP model and save the trained model to the saved_model/ directory. It also includes early stopping and learning rate reduction callbacks to optimize the training process.

3. Backtesting Simulation
You can use backtest_sim.py to simulate your strategy. It runs the model on historical data and evaluates performance based on specified metrics.

python utils/backtest_sim.py
4. Model Analysis
After training the model, you can analyze its performance using the following:

python utils/analyse_model.py
This script loads the saved model and provides insights into its architecture and performance.

Functions and Features
Data Processing (data_processing.py)
load_csv_data: Loads data from CSV files.
calculate_average: Calculates the average of two DataFrames.
log_transform_column: Applies log transformation to a given column.
add_time_features: Adds time-based features such as year, month, day, hour, and minute.
calculate_percentage_change: Calculates percentage change relative to the open price.
Indicators (indicator.py)
calculate_ema: Calculates Exponential Moving Average (EMA) for a given period.
calc_up_and_down: Calculates up and down thresholds based on price and volume data.
Model (MLP.py)
build_mlp_model: Defines and compiles an MLP model for regression or classification tasks. The model includes input layers, hidden layers, and output layers with appropriate activation functions (ReLU, Softmax, etc.).
Training and Saving Model (train_and_save_model_main.py)
Preprocesses data and trains the MLP model.
Saves the trained model to the saved_model/ directory.
Dependencies
This project relies on the following Python libraries:

tensorflow for model building and training
pandas for data manipulation
numpy for numerical computations
sklearn for data splitting and evaluation
matplotlib for plotting and visualizations
You can install the required dependencies using:

pip install -r requirements.txt
License
This project is licensed under the MIT License.

Contributing
Feel free to fork this repository and submit pull requests for new features or bug fixes.
This `README.md` provides a detailed overview of your project, including usage instructions, functionality explanations, and the project structure. It covers how to load data, preprocess it, train a model, and evaluate its performance. &#8203;:contentReference[oaicite:0]{index=0}&#8203;
