# Kaggle Titanic Survival Prediction - 0.80 Milestone Solution

This repository features a highly effective solution for the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic), consistently scoring **0.80+**.

## Key Strategies

1. **Group Survival Feature (The Game Changer)**
   Instead of looking at passengers in isolation, this solution tracks survival status across families and groups. By linking passengers via `Surname`, `Fare`, and `Ticket` numbers, it effectively captures the "women and children first" protocol as it applied to entire travel groups.

2. **Advanced Preprocessing**
   - **Grouped Age Imputation**: Missing ages are filled based on the median age of specific `Title` and `Pclass` groups.
   - **Integer Binning**: Age and Fare are transformed into ordinal bins. By converting these to integers, we eliminate common `ValueError` issues associated with XGBoost and categorical dtypes.

3. **Weighted Ensemble**
   A `VotingClassifier` combines Random Forest, Gradient Boosting, and XGBoost. We employ soft voting with weights [1, 2, 2] to prioritize the stronger gradient-boosted models while maintaining stability through the random forest.

## Usage
Simply place `train.csv` and `test.csv` in the same folder as `titanic_solution.py` and run the script. The generated `submission.csv` is ready for Kaggle submission.