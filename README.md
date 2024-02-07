**Leveraging Historical Crash Data for Enhanced Insurance  Pricing**

Each year, there are approximately 6 million car accidents in the U.S.
About 3 million people are injured every year in car accidents, with 2 million of those experiencing permanent injuries. 
Over 36,000 people died in traffic crashes in 2019.

Our model reviews historical data to categorize accident severity
Classification informs resource allocation for claims processing

Modeling Approaches:
Logistic Regression
K-Nearest Neighbors (K-NN)
Decision Tree

Validation & Hyperparameter Tuning:
Technique: Nested Cross-Validation
Inner Loop: 5 folds (Hyperparameter tuning)
Outer Loop: 5 folds (Model performance)
Tuning Strategy: Grid Search

Evaluation Metric:
F1-macro (for multiclass classification)

Data Preparation:
Standardization of numeric & ordinal features
