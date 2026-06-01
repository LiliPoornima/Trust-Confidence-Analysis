# Trust vs Confidence in Healthcare – Statistical Modelling Analysis

## Project Overview

This project investigates the relationship between **patient trust in healthcare providers** and **confidence in the care they receive** using the CMS HCAHPS (Hospital Consumer Assessment of Healthcare Providers and Systems) dataset.

The study was conducted as part of the **Theory and Practices in Statistical Modelling (IT3011)** module under the **Data Science Specialization** at SLIIT.

### Research Statement

> "Patients who trust their healthcare providers feel more confident about their care."

The objective of this project is to validate this statement using descriptive, inferential, and predictive statistical modelling techniques.

---

## Dataset

Source: CMS HCAHPS Hospital Patient Satisfaction Data

The analysis utilizes multiple healthcare survey datasets containing:

- Patient satisfaction measures
- Communication quality indicators
- Hospital ratings
- Recommendation scores
- Response rates
- State and regional information

---

## Methodology

### 1. Data Preparation

- Data cleaning and preprocessing
- Missing value handling
- Dataset integration and merging
- Feature engineering
- Creation of composite Trust and Confidence scores

### 2. Descriptive Analytics

Performed statistical summarization of key variables:

- Mean
- Median
- Standard deviation
- Distribution analysis
- Trend analysis

#### Results

| Metric | Value |
|----------|----------|
| Mean Trust Score | 76.75 |
| Mean Confidence Score | 71.56 |
| Mean Response Rate | 26.74% |

---

### 3. Inferential Analytics

#### Pearson Correlation Analysis

Used to evaluate the strength and direction of the relationship between Trust and Confidence.

| Metric | Value |
|----------|----------|
| Correlation Coefficient (r) | 0.8681 |
| P-value | < 0.001 |

Interpretation:

- Strong positive relationship
- Statistically significant
- Null hypothesis rejected

---

#### Hypothesis Testing

##### Null Hypothesis (H₀)

There is no significant relationship between patient trust and confidence.

##### Alternative Hypothesis (H₁)

There is a significant relationship between patient trust and confidence.

##### Result

✅ Reject H₀

The analysis confirms a statistically significant relationship between Trust and Confidence.

---

#### Independent T-Test

Compared confidence levels between High-Trust and Low-Trust groups.

| Metric | Value |
|----------|----------|
| T-statistic | 18.85 |
| P-value | < 0.001 |

Result:

Patients with higher trust levels exhibit significantly higher confidence scores.

---

#### ANOVA

Evaluated differences in confidence scores across regions.

| Metric | Value |
|----------|----------|
| F-statistic | 45.06 |
| P-value | < 0.001 |

Result:

Significant regional differences were observed.

---

### 4. Predictive Analytics

#### Simple Linear Regression

Modelled the impact of Trust on Confidence.

#### Multiple Linear Regression

Included:

- Trust Score
- Response Rate

to predict Confidence Score.

---

## Model Performance

| Metric | Value |
|----------|----------|
| R² Score | 0.7537 |
| RMSE | 1.9017 |
| Trust Coefficient | 0.8798 |
| Response Rate Coefficient | 0.1173 |

### Interpretation

- Trust is a strong predictor of confidence.
- Approximately 75.4% of the variation in Confidence Score is explained by the model.
- Higher trust levels are associated with increased patient confidence.

---

## Visualizations

The project generates the following visualizations:

- Distribution plots
- Box plots
- Correlation heatmap
- Regression plot
- Residual plot
- Trend analysis
- State-level scatter plots
- ANOVA mean comparison plot
- T-test comparison plot

Generated figures are automatically saved in:

```
outputs/figures/
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Key Findings

- Strong positive correlation between Trust and Confidence (r = 0.8681).
- Statistical testing confirmed a significant relationship (p < 0.001).
- Trust is a meaningful predictor of patient confidence.
- Patients with higher trust levels consistently report higher confidence in healthcare services.
- Communication quality and responsiveness play an important role in shaping patient perceptions.

---

## Conclusion

This study successfully validates the research statement:

> "Patients who trust their healthcare providers feel more confident about their care."

Using descriptive analytics, inferential statistics, and predictive modelling, the findings demonstrate that trust is a significant factor influencing patient confidence and perceived healthcare quality.

---

## Team Members

- Durangi Abeykoon
- Theekshana Ranasinghe
- Poornima Liyanage
- Lahiruni Ariyawansa

---

## Academic Context

**Module:** Theory and Practices in Statistical Modelling (IT3011)

**Specialization:** Data Science

**Institution:** SLIIT
