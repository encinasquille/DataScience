# 🧪 Glass Identification Dataset

The **Glass Identification Dataset** is a classic and widely referenced repository in the machine learning and pattern recognition literature. Developed within the context of forensic investigations, the primary objective of this dataset is to assist in the classification and identification of different origins of glass fragments based on their physicochemical composition.

The original motivation for its development lies in forensic science: at a crime scene, accurately identifying the type of glass of a recovered fragment can provide crucial evidence linking it to a scene or suspect (e.g., differentiating structural building window glass from vehicle headlamp glass) (Evett & Spiehler, 1987).

---

## 📊 Dataset Description

The dataset comprises **214 independent observations**, characterized by **10 continuous numeric variables**. The conceptual structure of the variables is divided as follows:

- **9 continuous attributes (independent variables/features):** Representing chemical and thermodynamic properties (refractive index and weight percent composition of metallic oxides).
- **1 nominal categorical variable (dependent/target variable):** Representing the functional class or final application of the glass, denoted as `Type`.

### 🔢 Predictor Attributes (Features)

The elemental composition is measured through the weight percentage of the element in its corresponding oxide form.

| Variable | Scientific Description | Unit / Nature |
|:--------:|:-----------------------|:--------------|
| **RI** | Refractive Index | Dimensionless |
| **Na** | Sodium Content | Weight % in oxide |
| **Mg** | Magnesium Content | Weight % in oxide |
| **Al** | Aluminum Content | Weight % in oxide |
| **Si** | Silicon Content | Weight % in oxide |
| **K**  | Potassium Content | Weight % in oxide |
| **Ca** | Calcium Content | Weight % in oxide |
| **Ba** | Barium Content | Weight % in oxide |
| **Fe** | Iron Content | Weight % in oxide |

### 🎯 Target Variable

The output variable maps the technological manufacturing process and final application of the glass into **6 active classes** (Class 4 has no sampled instances in this dataset):

| Class (`Type`) | Functional Category | Process Description |
|:--------------:|:--------------------|:--------------------|
| **1** | Building Windows | *Float Processed* (Continuous processing over molten tin) |
| **2** | Building Windows | *Non-Float Processed* |
| **3** | Vehicle Windows | *Float Processed* |
| **5** | Containers | General packaging glass (bottles, jars) |
| **6** | Tableware | Glass for varied domestic utility |
| **7** | Headlamps | Automotive optics |

---

## 🧠 Problem Formulation

From a methodological standpoint, the challenge is framed as a **supervised multi-class classification problem**. Through statistical predictive modeling methodologies, the primary objective is:

> To establish a probabilistic or deterministic mapping $f: X \rightarrow y$, where the physicochemical feature vector $X \in \mathbb{R}^9$ is utilized to accurately infer the functional glass class $\hat{y} \in \{1, 2, 3, 5, 6, 7\}$.

### 🔍 Statistical Considerations and Intrinsic Challenges

The scientific literature points out rigorous characteristics that demand meticulous modeling when approaching this dataset:

1. **Class Imbalance:** There is a strong predominance of classes related to structural residential building glass, with a scarcity of marginal data (e.g., tableware and headlamps).
2. **Non-Gaussian Distributions:** Some trace elements, structurally speaking like Barium (`Ba`) and Iron (`Fe`), present considerable positive skewness alongside a high frequency of zero-values, demanding adequate data transformations.
3. **Multicollinearity:** Thermo-physical phenomena condition a strong theoretical covariance, where for example, the refractive index (RI) correlates significantly with Calcium (`Ca`) and Silica (`Si`) compositions, necessitating multivariate or structural regularization approaches (L1/L2 penalties).

---

## 📈 Exploratory Data Analysis (EDA) Methodology

A primary methodological evaluation studying this phenomenological data suggests:

- **Correlational and Multivariate Analysis:** Parametric and non-parametric application (*Pearson's r*, *Spearman's ρ*, and the *Phi-K* $\phi_{K}$ coefficient) to evaluate linear and non-linear interactions associated with the prediction.
- **Dimensionality Reduction:** Techniques such as Principal Component Analysis (PCA) or t-SNE can mitigate collinear artifacts mentioned above.
- **Hypothesis Testing:** To detect compositional significance across neighboring class boundaries (e.g., Float vs. Non-Float processed).

### ⚙️ Computational Experiment Application (Python)

A verification of univariate relationships focusing on material type using a coefficient that accommodates non-linearities (`Phi-K`):

```python
import pandas as pd
from phik import phik_matrix

# Ingesting the glass dataset
df = pd.read_csv("glass.csv")

# Building the Phi-K statistical correlation matrix
phik_corr = df.phik_matrix()

# Extracting the marginal relative importance of features against the dependent variable
target_corr = phik_corr["Type"].sort_values(ascending=False)
print("Feature correlation (Phi-K) versus Target variable 'Type':")
print(target_corr)
```

---

## 📚 References

1. **Evett, I. W., & Spiehler, E. J. (1987).** _Rule-based expert systems for forensic glass classification._ In *Expert Systems in Law*, ed. A.W.F. Huggins, vol. 12, pp. 1-13. (Original publication introducing these methodologies into forensic sciences).
2. **German, B. (1987).** _Glass Identification Database_. Central Research Establishment, Home Office, Aldermaston, Reading, Berkshire (UK).
3. **Dua, D., & Graff, C. (2019).** _UCI Machine Learning Repository_ [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 
   - Dataset access: [Glass Identification Data Set](https://archive.ics.uci.edu/dataset/42/glass+identification)
   - Donor: **Vina Spiehler**, Ph.D., DABFT, Diagnostic Products Corporation.
