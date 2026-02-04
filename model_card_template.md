# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained to predict whether an individual's annual income exceeds $50,000 based on demographic and employment information from census data. The model was developed by Fabio Lima in February 2026 using the scikit-learn machine learning library.

The classifier uses an ensemble of 100 decision trees that collectively vote on predictions. Each tree is trained on a random subset of the data and features, which helps prevent overfitting and improves generalization. The random state is fixed at 42 to ensure reproducibility of results across different runs.

The trained model and its preprocessing artifacts are serialized using joblib and stored in the following files:

| File | Description |
|------|-------------|
| model/model.pkl | The trained RandomForestClassifier containing 100 decision trees |
| model/encoder.pkl | The fitted OneHotEncoder used to transform categorical features into numeric format |
| model/lb.pkl | The fitted LabelBinarizer used to convert salary labels between string and binary formats |

## Intended Use

This model is designed for educational purposes and serves as a demonstration of deploying a machine learning pipeline with FastAPI. It predicts whether a person earns more than $50,000 annually based on their demographic characteristics such as age, education, occupation, and work hours.

The primary users of this model are data scientists learning about ML deployment, researchers studying income prediction patterns, and students working on machine learning engineering projects. The model provides a baseline for understanding how census data can be used for income classification tasks.

This model should not be used for making actual decisions that affect individuals, such as hiring, lending, housing, or credit decisions. It is not suitable for production systems without thorough bias auditing and mitigation strategies. Since the training data comes exclusively from the United States Census, the model should not be applied to populations outside the United States.

## Training Data

The model was trained on the Census Income Dataset, also known as the Adult dataset, which originates from the UCI Machine Learning Repository. This dataset was extracted from the 1994 United States Census Bureau database and contains demographic information about individuals along with their income classification.

The complete dataset contains 32,562 records with 14 features and one target variable. For training, the data was split using an 80/20 ratio, resulting in approximately 26,050 samples for training and 6,513 samples for testing. The split was performed using scikit-learn's train_test_split function with a random state of 42 to ensure reproducibility.

The features include both continuous variables such as age, final weight, education years, capital gains, capital losses, and hours worked per week, as well as categorical variables including workclass, education level, marital status, occupation, relationship status, race, sex, and native country.

The target variable is salary, which has two classes: individuals earning $50,000 or less per year (labeled as <=50K) and those earning more than $50,000 per year (labeled as >50K). The dataset is imbalanced, with approximately 75% of samples belonging to the <=50K class and 25% belonging to the >50K class.

During preprocessing, categorical features are transformed using One-Hot Encoding with the handle_unknown parameter set to "ignore" to gracefully handle any previously unseen categories during inference. The target variable is binarized using a LabelBinarizer, converting <=50K to 0 and >50K to 1. Continuous features are not scaled since Random Forest classifiers are invariant to feature scaling.

## Evaluation Data

The evaluation data consists of 6,513 samples (20% of the original dataset) that were held out during training. These samples were processed using the same encoder and label binarizer that were fitted on the training data, which prevents data leakage and ensures that the evaluation reflects real-world inference conditions.

The test set maintains the same class distribution as the training set, with approximately 75% of samples in the <=50K class. This allows for an honest assessment of how the model will perform on new, unseen data from the same population.

## Metrics

The model is evaluated using Precision, Recall, and F1-Score, which are appropriate metrics for imbalanced classification problems. Accuracy is not used as the primary metric because a naive model that always predicts <=50K would achieve 75% accuracy while being completely useless for identifying high earners.

Precision measures the proportion of positive predictions that are actually correct. A precision of 0.74 means that when the model predicts someone earns >50K, it is correct 74% of the time. Recall measures the proportion of actual positive cases that were correctly identified. A recall of 0.64 means the model successfully identifies 64% of all people who actually earn >50K. The F1-Score is the harmonic mean of precision and recall, providing a single metric that balances both concerns.

### Overall Model Performance

Based on the evaluation on the test set, the model achieves the following approximate performance on the largest demographic groups:

| Metric | Score |
|--------|-------|
| Precision | 0.74 |
| Recall | 0.64 |
| F1-Score | 0.68 |

### Performance by Workclass

The model shows varying performance across different employment types. Federal government employees show the best performance with an F1-score of 0.7914, while self-employed individuals not incorporated show lower performance at 0.5789.

| Workclass | Precision | Recall | F1-Score | Count |
|-----------|-----------|--------|----------|-------|
| Federal-gov | 0.7971 | 0.7857 | 0.7914 | 191 |
| Self-emp-inc | 0.7807 | 0.7542 | 0.7672 | 212 |
| Local-gov | 0.7576 | 0.6818 | 0.7177 | 387 |
| State-gov | 0.7424 | 0.6712 | 0.7050 | 254 |
| Private | 0.7376 | 0.6404 | 0.6856 | 4,578 |
| Self-emp-not-inc | 0.7064 | 0.4904 | 0.5789 | 498 |
| ? (Unknown) | 0.6538 | 0.4048 | 0.5000 | 389 |

### Performance by Education Level

Education level significantly impacts model performance. The model performs best for individuals with advanced degrees and struggles with lower education levels.

| Education | Precision | Recall | F1-Score | Count |
|-----------|-----------|--------|----------|-------|
| Prof-school | 0.8182 | 0.9643 | 0.8852 | 116 |
| Doctorate | 0.8644 | 0.8947 | 0.8793 | 77 |
| Masters | 0.8271 | 0.8551 | 0.8409 | 369 |
| Bachelors | 0.7523 | 0.7289 | 0.7404 | 1,053 |
| HS-grad | 0.6594 | 0.4377 | 0.5261 | 2,085 |
| Some-college | 0.6857 | 0.5199 | 0.5914 | 1,485 |
| 7th-8th | 0.0000 | 0.0000 | 0.0000 | 141 |

### Performance by Sex

The model shows a notable performance gap between male and female individuals. Male individuals have an F1-score of 0.6997 compared to 0.6015 for female individuals, representing a difference of approximately 9.8 percentage points.

| Sex | Precision | Recall | F1-Score | Count |
|-----|-----------|--------|----------|-------|
| Male | 0.7445 | 0.6599 | 0.6997 | 4,387 |
| Female | 0.7229 | 0.5150 | 0.6015 | 2,126 |

### Performance by Race

Performance varies across racial groups. Asian-Pac-Islander individuals show the highest F1-score at 0.7458, while Amer-Indian-Eskimo individuals show the lowest at 0.5556 among groups with sufficient sample size.

| Race | Precision | Recall | F1-Score | Count |
|------|-----------|--------|----------|-------|
| Asian-Pac-Islander | 0.7857 | 0.7097 | 0.7458 | 193 |
| White | 0.7404 | 0.6373 | 0.6850 | 5,595 |
| Black | 0.7273 | 0.6154 | 0.6667 | 599 |
| Other | 1.0000 | 0.6667 | 0.8000 | 55 |
| Amer-Indian-Eskimo | 0.6250 | 0.5000 | 0.5556 | 71 |

### Performance by Occupation

Executive and managerial positions show the best performance with an F1-score of 0.7736, while farming and fishing occupations show lower performance at 0.3077.

| Occupation | Precision | Recall | F1-Score | Count |
|------------|-----------|--------|----------|-------|
| Exec-managerial | 0.7952 | 0.7531 | 0.7736 | 838 |
| Prof-specialty | 0.7880 | 0.7679 | 0.7778 | 828 |
| Tech-support | 0.7143 | 0.6863 | 0.7000 | 189 |
| Sales | 0.7273 | 0.6667 | 0.6957 | 729 |
| Farming-fishing | 0.5455 | 0.2143 | 0.3077 | 193 |
| Other-service | 1.0000 | 0.1923 | 0.3226 | 667 |

## Ethical Considerations

The model exhibits several concerning patterns that warrant careful consideration before any deployment. The performance disparity between male and female individuals, with a gap of nearly 10 percentage points in F1-score, suggests the model may perpetuate or amplify existing gender-based income disparities. Similarly, the lower performance for Amer-Indian-Eskimo individuals compared to other racial groups raises concerns about equitable treatment across demographics.

The training data originates from 1994, meaning it reflects income patterns and social conditions from over 30 years ago. Economic conditions, job markets, education requirements, and demographic compositions have changed substantially since then. Using this model to make predictions about current populations could introduce significant errors and potentially reinforce outdated stereotypes.

The model uses protected attributes including race and sex as input features. While these features may improve prediction accuracy, their use in income-related predictions is ethically problematic and may be legally restricted in many jurisdictions. Any deployment should carefully consider whether including these features aligns with applicable laws and ethical guidelines.

The Census data is anonymized and publicly available, but the model could potentially be misused to infer sensitive income information about individuals when combined with other data sources. Appropriate safeguards should be implemented to prevent such misuse.

## Caveats and Recommendations

The model has several important limitations that users should understand. First, the data is from 1994, and income patterns, job markets, and economic conditions have changed dramatically. The model should not be expected to accurately reflect current income distributions or determinants.

Second, the model achieves an F1-score of approximately 0.68, which means it incorrectly classifies a substantial portion of individuals. For high-stakes decisions, this error rate would be unacceptable. The model shows particularly poor recall, meaning it fails to identify many individuals who actually earn more than $50,000.

Third, the model performs poorly on certain subgroups. Individuals with lower education levels (7th-8th grade) have an F1-score of 0.00, meaning the model completely fails to identify high earners in this group. Similarly, certain occupations and demographic groups show substantially worse performance than others.

Fourth, the model uses default Random Forest parameters without hyperparameter tuning. Performance could likely be improved through grid search or random search optimization, though this would not address the fundamental limitations of the training data.

For anyone considering using this model, the following recommendations apply. Always review the slice_output.txt file to understand how the model performs across different demographic groups before making any deployment decisions. Do not use the model for consequential decisions affecting individuals without human oversight and robust appeal processes. If deployed in any capacity, retrain the model periodically with more recent data and monitor for distribution drift. Consider more sophisticated models such as gradient boosting or neural networks if improved performance is required. Implement fairness constraints or post-processing techniques to reduce performance disparities across protected groups before any deployment.

Future improvements should include hyperparameter tuning with cross-validation, feature importance analysis to understand which variables drive predictions, bias mitigation techniques such as reweighting or adversarial debiasing, updated training data from more recent census surveys, and model explainability tools like SHAP values or LIME to provide interpretable predictions.
