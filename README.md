# Osteoarthritis Treatment Analysis Using Machine Learning

## 📋 Project Overview

This project analyzes osteoarthritis treatment data to identify patterns in pain reduction across different treatment methods using clustering techniques and exploratory data analysis. The analysis aims to help healthcare professionals understand which treatments are most effective for different patient groups.

## 🎯 Problem Statement

**"Can we identify distinct patient clusters based on treatment response patterns and rank osteoarthritis treatments by effectiveness using machine learning clustering techniques?"**

## 📊 Dataset Information

- **Dataset Name:** OsteoarthritisData.csv
- **Analysis Focus:** Treatment effectiveness on pain scores
- **Key Variables:**
  - `y`: Pain score (primary outcome measure)
  - `se`: Standard error
  - `N`: Sample size
  - `time_wk`: Follow-up time in weeks
  - `treatname`: Treatment type
  - `AuthorDate`: Study publication date
  - `arm`, `narm`, `fupcount`, `fups`: Study design variables

## 🛠️ Technologies Used

- **Python **
- **Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` & `seaborn` - Data visualization
  - `warnings` - Warning management
- **powerBI**

## 📁 Project Structure

```
├── OsteoarthritisData.csv           # Original dataset
├── Cleaned_OsteoarthritisData.csv   # Processed dataset
├── osteoarthritis_analysis.py       # Main analysis script
├── README.md                        # Project documentation
└── outputs/
    ├── eda_correlation_heatmap.png
    ├── eda_distribution_y.png
    ├── eda_boxplot_treatment.png
    ├── elbow_method_plot.png
    └── cluster_pca_plot.png
```

## 🔍 Analysis Methodology

### 1. Data Preprocessing
- **Missing Value Handling:** Removed rows with null values
  * for loading the dataset*
  'df = pd.read_csv("OsteoarthritisData.csv")'
  * for checking for missing values*
  'print("Missing values:\n", df.isnull().sum())'
   <img width="287" height="452" alt="image" src="https://github.com/user-attachments/assets/023c5b07-f806-4c8a-9629-dd92a02cdc6c" />

- **Duplicate Removal:** Eliminated duplicate records
  'df = df.drop_duplicates()'
- **Outlier Detection:** Applied IQR method to remove outliers in pain scores

  " Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df['y'] >= lower) & (df['y'] <= upper)]
  
- **Feature Encoding:** One-hot encoded categorical variables (`treatname`, `AuthorDate`)
  " df_encoded = pd.get_dummies(df, columns=['treatname'], drop_first=True) "
  
- **Standardization:** Scaled numerical features using StandardScaler
  "scaler = StandardScaler()
numeric_cols = ['N', 'time_wk', 'y', 'se', 'arm', 'narm', 'fupcount', 'fups']
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])"

### 2. Exploratory Data Analysis (EDA)
- Generated descriptive statistics for all variables
  "print("Descriptive statistics:\n", df.describe())"
  <img width="1140" height="591" alt="image" src="https://github.com/user-attachments/assets/e00b8ce6-d85f-4a56-9796-53aff1c25ff8" />
  
- Created correlation heatmap to identify relationships
  "plt.figure(figsize=(10, 8))
   sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
   plt.title("Correlation Heatmap")
   plt.tight_layout()
   plt.savefig("eda_correlation_heatmap.png")
   plt.show()"
  <img width="1119" height="892" alt="image" src="https://github.com/user-attachments/assets/6605f508-5905-46c0-9896-3826b9b7a2b9" />

- Visualized pain score distribution
  "plt.figure(figsize=(8, 5))
   sns.histplot(df['y'], kde=True)
   plt.title("Distribution of Pain Score (y)")
   plt.tight_layout()
   plt.savefig("eda_distribution_y.png")
   plt.show()"
  <img width="1212" height="767" alt="image" src="https://github.com/user-attachments/assets/5ad99fde-99a5-488e-90a7-6d380e5e5070" />

- Analyzed treatment effectiveness using boxplots
  "plt.figure(figsize=(12, 6))
   sns.boxplot(data=df, x='treatname', y='y')
   plt.title("Pain Score by Treatment")
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.savefig("eda_boxplot_treatment.png")
   plt.show()"

  <img width="1315" height="671" alt="image" src="https://github.com/user-attachments/assets/349dcdb6-1e18-4ae6-bfb5-977a914cd56f" />


### 3. Machine Learning - Clustering Analysis
- **Algorithm:** K-Means Clustering
  "X = df_encoded[numeric_cols]
   kmeans = KMeans(n_clusters=3, random_state=42)
   df_encoded['Cluster'] = kmeans.fit_predict(X)"
  
- **Optimal Clusters:** Determined using Elbow Method
  "def plot_elbow_method(data, max_k=10):
    distortions = []
    K = range(1, max_k+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, marker='o')
    plt.title('Elbow Method - Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elbow_method_plot.png")
    plt.show()

# Run on your scaled features
plot_elbow_method(X)"
<img width="1187" height="738" alt="image" src="https://github.com/user-attachments/assets/c9a403c3-b084-4fb9-891c-00a0d4c354ae" />

- **Evaluation Metric:** Silhouette Score
  "score = silhouette_score(X, df_encoded['Cluster'])
   print(f"Silhouette Score: {score:.4f}")"
  
- **Visualization:** PCA-based 2D cluster visualization
  "from sklearn.decomposition import PCA

# Reduce features to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot with cluster labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_encoded['Cluster'], palette='Set1')
plt.title('Cluster Visualization (PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.tight_layout()
plt.savefig("cluster_pca_plot.png")
plt.show()"
<img width="1186" height="863" alt="image" src="https://github.com/user-attachments/assets/7026585b-4ddc-4a68-89fd-24a1e53e4384" />

**cleaned dataset**
f_encoded.to_csv("Cleaned_OsteoarthritisData.csv", index=False)
<img width="1485" height="669" alt="image" src="https://github.com/user-attachments/assets/5991a3a7-ad22-4772-8214-025938ae2115" />
<img width="1007" height="875" alt="image" src="https://github.com/user-attachments/assets/8142ab8b-b825-4cad-95fa-5dbb86da21cc" />




### 4. Custom Analysis Functions
- `cluster_summary()`: Analyzes mean pain scores by cluster and treatment
  def cluster_summary(df_with_clusters, original_df):
    temp = df_with_clusters.copy()
    temp['original_treatname'] = original_df['treatname'].values
    summary = temp.groupby(['Cluster', 'original_treatname'])['y'].mean().unstack()
    return summary

# Use the function
summary_table = cluster_summary(df_encoded, df)
print("Mean Pain Score by Cluster and Treatment:\n", summary_table)
<img width="1084" height="769" alt="image" src="https://github.com/user-attachments/assets/9bc28ad4-f9c4-4d93-be9f-0c9955255f5e" />

  

- `rank_treatments_by_cluster()`: Ranks treatments by effectiveness within each cluster
  
   "def rank_treatments_by_cluster(df_with_clusters, original_df):
    df_temp = df_with_clusters.copy()
    df_temp['treatname'] = original_df['treatname'].values
    
    # Compute average pain score (y) for each treatment within each cluster
    cluster_treatment_means = df_temp.groupby(['Cluster', 'treatname'])['y'].mean().reset_index()

    # Rank treatments within each cluster (lower pain score is better)
    cluster_treatment_means['Rank'] = cluster_treatment_means.groupby('Cluster')['y'].rank(method='dense')
    
    return cluster_treatment_means.sort_values(['Cluster', 'Rank'])

# --- Applying the function and view result ---
treatment_rankings = rank_treatments_by_cluster(df_encoded, df)
print(treatment_rankings)"
<img width="584" height="431" alt="image" src="https://github.com/user-attachments/assets/2b3825e4-e2bc-4d3a-8f52-7861d046d94c" />

- `plot_elbow_method()`: Determines optimal number of clusters
  def plot_elbow_method(data, max_k=10):
    distortions = []
    K = range(1, max_k+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, marker='o')
    plt.title('Elbow Method - Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elbow_method_plot.png")
    plt.show()

# Run on your scaled features
plot_elbow_method(X)
<img width="1194" height="741" alt="image" src="https://github.com/user-attachments/assets/51a01a4c-8b29-41f8-abc4-88d29ba0013d" />


## 📈 Key Findings

### Clustering Results
- **Number of Clusters:** 3 optimal clusters identified
  <img width="136" height="128" alt="image" src="https://github.com/user-attachments/assets/c33b8ddc-38e2-4402-9e11-0ba00b74e86b" />

- **Silhouette Score:**
  <img width="334" height="40" alt="image" src="https://github.com/user-attachments/assets/adb49c13-0326-4036-888a-d196316e1e8e" />

  
- **Patient Segmentation:** Successfully grouped patients with similar treatment response patterns
  <img width="594" height="427" alt="image" src="https://github.com/user-attachments/assets/4a839a91-c9c8-4377-b8db-844ddc788c29" />


### Treatment Rankings
- Treatments ranked by effectiveness within each patient cluster
  <img width="594" height="427" alt="image" src="https://github.com/user-attachments/assets/4a839a91-c9c8-4377-b8db-844ddc788c29" />
  
- Lower pain scores indicate better treatment outcomes
- Cluster-specific treatment recommendations generated

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution Steps
1. Clone this repository
2. Ensure `OsteoarthritisData.csv` is in the project directory
3. Run the analysis script:
   ```python
   python osteoarthritis_analysis.py
   ```
4. Check the `outputs/` folder for generated visualizations
5. Review `Cleaned_OsteoarthritisData.csv` for processed data

## 📊 Generated Outputs

### Visualizations
1. **Correlation Heatmap** - Shows relationships between variables
2. **Pain Score Distribution** - Histogram with KDE curve
3. **Treatment Boxplot** - Pain scores by treatment type
4. **Elbow Method Plot** - Optimal cluster determination
5. **PCA Cluster Visualization** - 2D representation of clusters

### Data Files
- **Cleaned Dataset** - Preprocessed and ready for further analysis
- **Cluster Assignments** - Each patient assigned to optimal cluster

## 🎯 Business Impact

### Healthcare Applications
- **Personalized Treatment:** Identify which treatments work best for specific patient groups
- **Resource Allocation:** Focus resources on most effective treatments
- **Clinical Decision Support:** Data-driven treatment recommendations

### Research Implications
- **Treatment Comparison:** Objective ranking of treatment effectiveness
- **Patient Stratification:** Better understanding of patient response patterns
- **Future Studies:** Framework for analyzing treatment outcomes

## 🔮 Future Enhancements

- **Advanced Modeling:** Implement supervised learning for treatment prediction
- **Time Series Analysis:** Analyze treatment effectiveness over time
- **Feature Engineering:** Create composite scores for better clustering
- **Interactive Dashboard:** Develop web-based visualization tool
- **External Validation:** Test findings on additional datasets

## 📝 Model Performance

- **Clustering Algorithm:** K-Means with optimal k=3
- **Evaluation Method:** Silhouette analysis and visual inspection
- **Data Quality:** Comprehensive preprocessing with outlier removal
- **Reproducibility:** Fixed random state for consistent results

## 👥 Usage Guidelines

### For Healthcare Professionals
- Review cluster characteristics to understand patient groups
- Use treatment rankings for evidence-based decision making
- Consider cluster membership when selecting treatments

### For Researchers
- Extend analysis with additional variables
- Validate findings with external datasets
- Apply methodology to other treatment studies

## 📧 Contact Information

For questions about this analysis or collaboration opportunities, please reach out through the repository issues section.

## 🙏 Acknowledgments

- Dataset contributors and researchers in osteoarthritis treatment
- Open-source community for providing excellent tools and libraries
- Healthcare professionals providing domain expertise

---

**Note:** This analysis is for educational and research purposes. Clinical decisions should always involve qualified healthcare professionals and consider individual patient circumstances.
