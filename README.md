# Osteoarthritis-Treatment-Analysis-Project

## Project Overview
This project analyzes clinical trial data for osteoarthritis treatments to:
- Identify patterns in pain reduction across different treatments
- Cluster patients based on treatment response
- Recommend optimal treatments for specific patient groups

## Dataset
**OsteoarthritisData.csv**  
Contains clinical trial records with:
- Treatment details (`treatname`)
- Pain scores (`y`)
- Study characteristics (`N`, `time_wk`, `se`, etc.)

## Data Cleaning Process
```python
# Sample cleaning code
df = pd.read_csv("OsteoarthritisData.csv")
df = df.drop_duplicates().dropna()
df['y'] = df['y'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)  # Outlier removal
**missing values
<img width="783" height="429" alt="image" src="https://github.com/user-attachments/assets/ad2611c6-b61e-41a4-9363-0d9e67176429" />
**descriptive statistics
'print("Descriptive statistics:\n", df.describe())'
<img width="1040" height="574" alt="image" src="https://github.com/user-attachments/assets/8b132ace-a824-4bfd-9af6-328ecd4023ef" />




