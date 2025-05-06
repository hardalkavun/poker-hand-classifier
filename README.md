
# Poker Hand Classification 

This project uses the [UCI Poker Hand dataset](https://archive.ics.uci.edu/ml/datasets/poker+hand) to classify poker hands using various machine learning algorithms.

## Project Structure

Each algorithm is implemented as a separate module:

- `naive_bayes/` — Gaussian Naive Bayes
- `logistic_regression/` — Logistic Regression
- `decision_tree/` — Decision Tree
- `random_forest/` — Random Forest
- `xgboost/` — XGBoost
- `linear_regression/` — Linear Regression (note: not ideal for categorical targets)
- `neural_network/` — Artificial Neural Network (MLP)

## Feature Engineering

The following features are engineered from the raw dataset:

- **is_flush**: Whether all cards have the same suit
- **is_straight**: Whether cards form a sequence
- **hand_type**: Encoded poker hand type (e.g., pair, two pair, three of a kind)
- **high_card**: The highest card in the hand

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/hardalkavun/poker-hand-classification.git
cd poker-hand-classification
pip install -r requirements.txt
