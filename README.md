# Power of Plots using Matplotlib

## Project Overview

* Analyze impact of 4 drug treatments performed on 250 mice identified with SCC tumor growth over 45 days of clinical trial.
* Leverage power of plots generated using Matplotlib libraries to perform statistical analysis and derive observations.


## Dependencies

Installation requires [numpy](https://numpy.org), [scipy](https://www.scipy.org), [pandas](https://pandas.pydata.org),[Matplotlib](https://matplotlib.org) and [Jupyter Notebook](https://jupyter.org/install).

## Analysis

* Capomulin outperformed Infubinol and Ceftamin by a significant margin. However Ramicane performed slightly better than Capomulin with average tumor size 34.84 as compared to 37.31 of Capomulin.

* Mouse s185 treated with Capomulin showed 51% reduction in tumor volume, however a slight increase in tumor volume was observed between 10 and 15 days of the treatment. This might imply minimal treatmeant period necessary to see sustained results.

* Average tumor volume increased exponentially with increase in mouse weight. This might imply need for higher drug dosage with increase in weight.

## Execution

### Dependencies and starter code

```import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
import seaborn as sns
from scipy.stats import linregress

# Study data files
mouse_metadata = "data/Mouse_metadata.csv"
study_results = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata)
study_results = pd.read_csv(study_results)

# Combine the data into a single dataset
combined_df = pd.merge(mouse_metadata,study_results,how='outer',on='Mouse ID')
combined_df.to_csv("data/combined.csv", index=False, header=True) 
```

### Summary statistics of the tumor volume for each drug regimen.

```
summary_stats = combined_df.groupby(['Drug Regimen']).agg({'Tumor Volume (mm3)': ['mean','median','var','std','sem']})
summary_stats.columns = ['Mean', 'Median', 'Variance','Standard Deviation','Standard Error of Mean']

summary_stats.head()
```

| Drug Regimen  |  Mean         | Median    | Variance  | Standard Deviation| Standard Error of Mean|
| ------------- |:-------------:| ---------:| ---------:| -----------------:| ---------------------:|
| Capomulin     | 40.675741     | 41.557809 | 24.947764 | 4.994774          | 0.329346              |
| Ceftamin      | 52.591172     | 51.776157 | 39.290177 | 6.268188          | 0.469821              |
| Infubinol     | 52.884795     | 51.820584 | 43.128684 | 6.567243          | 0.492236              |
| Ketapril      | 55.235638     | 53.698743 | 68.553577 | 8.279709	        | 0.603860              |
| Naftisol      | 54.331565     | 52.509285 | 66.173479 | 8.134708          | 0.596466              |


### Bar plot using Matplotlib's pyplot that shows number of data points for each treatment regimen.

```
regimen_count = combined_df['Drug Regimen'].value_counts()
regimen = [reg for reg in combined_df['Drug Regimen'].unique()]

plt.bar(regimen, regimen_count,linewidth=1,align="center",width=0.5)
plt.xticks(rotation=90)
plt.ylim(0, max(regimen_count)+10)

plt.legend("Drug Count")
plt.title("Drug Count Per Regimen")
plt.xlabel("Drug Regimen")
plt.ylabel("Drug Count")
```
![](Images/drug_count_per_regimen.png)

### Pie plot using Matplotlib's pyplot that shows the distribution of female/male mice in the study.

### Calculate final tumor volume of each mouse across four of the most promising treatment regimens: Capomulin, Ramicane, Infubinol, and Ceftamin. Calculate the quartiles and IQR and quantitatively determine if there are any potential outliers across all four treatment regimens.

### Using Matplotlib, generate a box and whisker plot of the final tumor volume for all four treatment regimens and highlight any potential outliers in the plot by changing their color and style.

### Generate a line plot of time point versus tumor volume for a single mouse treated with Capomulin.

### Generate a scatter plot of mouse weight versus average tumor volume for the Capomulin treatment regimen.

### Calculate the correlation coefficient and linear regression model between mouse weight and average tumor volume for the Capomulin treatment. Plot the linear regression model on top of the previous scatter plot.






plt.savefig("Images/drug_count_per_regimen.png")

   
