# Exoplanet Detection with Machine Learning & TESS Data

```python
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
```

# 1. About the Data & Preprocessing

The data we are going to use consist of two parts <br>
TESS Input Catalog and MAST Archive 

### 1.1. TESS Input Catalog 
This list contains the star names (tid) and the labels for the classifier (tfopwg_disp) <br>
Downloaded through Nasa Exoplanet Archive.<br>
Link: <a>https://exoplanetarchive.ipac.caltech.edu/index.html</a>



```python
import pandas as pd
df = pd.read_csv("toi_data.csv")
df = df[["tid", "tfopwg_disp"]]
df = df[df["tfopwg_disp"].isin(["CP", "FP"])].copy()
df["tfopwg_disp"].value_counts()
```




    tfopwg_disp
    FP    1228
    CP     705
    Name: count, dtype: int64



### 1.2. MAST Archive ve Lightkurve
The lightcurve data is in MAST (Mikulski Archive for Space Telescopes) and we are using Python Lightcurve library to download and process. <br>
As an example we are going to focus on Pi Mensae C. A clean and confirmed exoplanet signal. <br>


```python
import lightkurve as lk
# parameters
# id for Pi Mensae C
# mission = TESS data
# SPOC preprocessed and cleaned data by NASA
tpf = lk.search_targetpixelfile("TIC 261136679", mission="TESS", author="SPOC").download()

# Plot the pixels
tpf.plot()
plt.show()  
```


    
![png](analysis_files/analysis_5_0.png)
    


The data we need is the flux value. <br>
Transit method: The planet blocks some of the host stars light while passing between the telescope and the host star. <br> 
This translates to a lower flux value on the telescope instruments.


```python

# download the lightcurve
search_result = lk.search_lightcurve("TIC 261136679", mission="TESS", author="SPOC") # 261136679
lc = search_result.download()
lc.plot()
plt.show()
```


    
![png](analysis_files/analysis_7_0.png)
    


The data spans around 27 days. This is because the orbit of the telecope around Earth. Also it sends data once every 13 days. This is the reason for the gaps in the data.

The noise in the data is instrumental and environmental noise. <br>  
The telecope sensitivity, orbit around Earth, wobbling and vibrating, background stars are contributing factors to this noise.

We first remove the empty data.<br>
Then we pick the best quality data that is flagged by nasa. <br>
We remove outliers.

The light coming from the stars are irregular and non-static. It fluctuates greatly over time. <br>
So we flatten the lightcurve to correct for this fluctuation using .flatten().

Flatten applies Savitzky-Golay filter to smooth out the data.
We are trying to reduce the noise while keeping the shape of the data.

https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay



```python

lc = lc.remove_nans() 
lc = lc[lc.quality == 0] # quality control
lc = lc.remove_outliers(sigma = 3) # remove the observations that are 3 sigmas away from the mean
# parameters
# window_length: number of points used for moving average calculation
# why 401 is chosen: it is a rule of thumb with the TESS data
# this is small enough to remove long-term trends but large enough not to remove the transit signal
flat_lc = lc.flatten(window_length=401)
flat_lc.scatter()
plt.show()  
```


    
![png](analysis_files/analysis_9_0.png)
    


The next step is to search for a dip in the data <br>
We need to know the transit period of the planet <br>
We use bls algorightm to find it <br>
The period of the host star is 6.27 but the algorithm will find it for us. <br> 


```python
import numpy as np
pg = lc.to_periodogram(method="bls", period=np.linspace(1, 15, 5000)) # box least squares
pg.plot()
plt.show()
```


    
![png](analysis_files/analysis_11_0.png)
    


Periyodu bulduktan sonra periyodun ilk keşfedildiği zamandan başlayarak veriyi "katlamamız" gerekiyor.<br>
We need to "fold" data from the transit period the BLS algorithm found. <br>
We also bin the data to reduce noise. <br>
This is what we feed into the classifier model along with statistical features of the lightcurve. <br>
This way the transit signals will overlap with itself and the transit will be more visible. <br>


```python
best_period = pg.period_at_max_power.value # Best period
best_t0 = pg.transit_time_at_max_power.value # Best epoch time (start of the period)
folded_lc = lc.fold(period=best_period, epoch_time=best_t0)
folded_lc = folded_lc.bin(bins=50) # binning to reduce noise
folded_lc.plot()
plt.show()
```


    
![png](analysis_files/analysis_13_0.png)
    


# 2. Feature Extraction
We choose the below statistical features alongside the flux data.<br>
Standard deviation, <br>
Skewness, must be negative in the exoplanet signal <br>
Kurtosis <br>

Coming from the BLS algorithm:<br>
Period,<br>
Transit depth (derinlik), indicates how much the exoplanet blocks the host stars light. This the feature used to estimate the size of the exoplanet. <br>
Max power, the peak point in the BLS periodogram, power of the signal<br>

This is what the data looks like after the feature extraction.


```python
import pandas as pd
features = pd.read_pickle("extracted_features.pkl")
# display first 9 columns and last 2 columns
features = features.iloc[:, list(range(9)) + [-2, -1]]
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>std_dev</th>
      <th>skewness</th>
      <th>kurtosis</th>
      <th>period</th>
      <th>transit_depth</th>
      <th>transit_duration</th>
      <th>max_power</th>
      <th>flux_0</th>
      <th>flux_1</th>
      <th>flux_49</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001820</td>
      <td>0.003804</td>
      <td>-0.168127</td>
      <td>12.409482</td>
      <td>0.000235</td>
      <td>0.05</td>
      <td>2.205922</td>
      <td>0.999995</td>
      <td>0.9999998166223942</td>
      <td>1.000003</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001591</td>
      <td>0.010904</td>
      <td>-0.147384</td>
      <td>1.868174</td>
      <td>0.000093</td>
      <td>0.05</td>
      <td>4.580811</td>
      <td>0.999998</td>
      <td>0.9999938580887549</td>
      <td>0.999983</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000728</td>
      <td>0.054569</td>
      <td>-0.118460</td>
      <td>2.744749</td>
      <td>0.000280</td>
      <td>0.10</td>
      <td>40.896649</td>
      <td>0.999960</td>
      <td>1.0000520450597414</td>
      <td>0.999968</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.001402</td>
      <td>0.001151</td>
      <td>-0.179032</td>
      <td>11.373275</td>
      <td>0.000145</td>
      <td>0.05</td>
      <td>2.169721</td>
      <td>1.000006</td>
      <td>0.9999923744428908</td>
      <td>1.000003</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.001545</td>
      <td>0.006737</td>
      <td>-0.117861</td>
      <td>4.551110</td>
      <td>0.000172</td>
      <td>0.10</td>
      <td>13.378276</td>
      <td>0.999997</td>
      <td>1.0000123763028184</td>
      <td>1.000004</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Exculuded some flux features for better visibility. There are 50 flux features starting from 0 to 49.

# 3. Fine Tuning
We used Randomized Search for fine tuning. <br>
We remove nans and non-numerical data as best practice. <br>
Empty observations are filled with mean using simple imputer. <br>

<b>Chosen Parameter Distributions</b>

<img src="results\param_distros.png" alt="2" style="width: 500px">
<img src="results\param_distros2.png" alt="2" style="width: 700px">


### 3.1. Class Imbalance
Exoplanet transits occur very rarely due to the physical constraints of the TESS telescope. <br>
The maximum planetary orbital period observable via the transit method using TESS is 13.5 days.<br>
However, as we know from our own solar system, even the planet closest to our star has a period of 88 days.<br>
Due to this limitation and the prevalence of binary star systems, there is an imbalance in the dataset.<br>
This imbalance was addressed within the pipeline using the SMOTE (Synthetic Minority Over-sampling Technique) method, by simulating minority observations exclusively in the training set.


```python
# Class distribution 0: 984, 1: 692
# Approximately 58.7% negative (0) and 41.3% positive (1)
features["target"].value_counts()
```




    target
    0    984
    1    692
    Name: count, dtype: int64



### 3.2 Randomized Search

Randomized search için 30 hiperparametre kombinasyonu, <br>
30 hyper-parameter combinations for Randomized Search <br>
5-fold Cross Validation, <br>
F1 Score for performance measurements.

<b> Pipeline </b>

<img src="results\pipeline.png" alt="2" style="width: 700px">

### 3.3. Results and Decision Treshold Optimisation

"Due to the class imbalance in the dataset, the default decision threshold of 0.5 is insufficient for the problem at hand. When generating predictions, a model typically selects the class with a probability exceeding 0.5; this cut-off point is known as the decision threshold.

Imagine you are the scientist overseeing the machine learning model in this project. <br>
If a light curve is fed into a model with a standard 0.5 threshold and the model assigns it a positive probability of 0.4, it will predict a 'negative' result, causing you to discard that potential candidate. However, if that light curve actually has a real planet, that discovery is effectively lost to science until a future telescope observation occurs. <br> 
A 40% probability is actually quite significant for an exoplanet candidate, and from a scientific standpoint, this is a possibility we cannot afford to miss. 

Therefore, the decision thresholds of the best models trained via Randomized Search were processed through an optimization algorithm to identify and update the optimal threshold for each model. 

The final performance results obtained following decision threshold optimization are as follows:" 



```python
results = pd.read_csv("results/final_performance_report.csv")
best_params = results["Best Params"]
results.drop(columns=["Best Params"], inplace=True)
results.to_excel("results/final_performance_report.xlsx", index=False)
results

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Best CV Score (F1)</th>
      <th>Best Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>New F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest</td>
      <td>0.628</td>
      <td>0.48</td>
      <td>0.630</td>
      <td>0.713</td>
      <td>0.669</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVM</td>
      <td>0.618</td>
      <td>0.40</td>
      <td>0.516</td>
      <td>0.902</td>
      <td>0.657</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gradient Boosting</td>
      <td>0.611</td>
      <td>0.44</td>
      <td>0.540</td>
      <td>0.828</td>
      <td>0.654</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNN</td>
      <td>0.612</td>
      <td>0.54</td>
      <td>0.564</td>
      <td>0.762</td>
      <td>0.648</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Logistic Regression</td>
      <td>0.609</td>
      <td>0.50</td>
      <td>0.526</td>
      <td>0.820</td>
      <td>0.641</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Decision Tree</td>
      <td>0.596</td>
      <td>0.10</td>
      <td>0.445</td>
      <td>0.967</td>
      <td>0.610</td>
    </tr>
  </tbody>
</table>
</div>



### 3.4 Best Parameters of the First Model


```python
import pprint
pprint.pprint(best_params[0], width=40)
```

    ("{'model__bootstrap': True, "
     "'model__criterion': 'gini', "
     "'model__max_depth': 23, "
     "'model__max_features': None, "
     "'model__min_samples_leaf': 9, "
     "'model__min_samples_split': 16, "
     "'model__n_estimators': 114}")
    

### 3.5. Feature Importance

<img src="results/shap_beeswarm.png" alt="2" style="width: 500px">

We measured feature importance with SHAP.

Here, the feature "flux_28" stands out as the model's most strongly discriminating feature. This feature corresponds to the 28th bin in the dataset, which was binned to reduce noise following the data folding process. After the period is determined using the BLS method, the folding algorithm aligns the planetary signal to be centered within the light curve. While the transit signal is strongest around bins 24 and 26, the greatest contribution to the decision comes from the 28th bin. When this specific bin is plotted, we can see that the values cluster around 1.

<img src="results\flux_28_analysis_train_data.png" alt="2" style="width: 500px">

So, while the 28th bin is located near the center of the folded light curve, the values it contains present a somewhat counterintuitive result. This indicates that the Random Forest model analyzes not only scalar statistics—such as depth, period, and transit duration—but also the morphology of the transit itself. Here, we identify the 'period' as the model's second strongest discriminating feature. This highlights that a fundamental characteristic of planetary transit signals is their repetition at regular and consistent intervals, as opposed to random noise. The model utilized this periodicity, detected by the BLS algorithm, as a primary filter to distinguish exoplanets from instrumental errors and noise.

### 3.6. Confusion Matrix

Random Forest model yields the best overall score; however, the Support Vector Machine (SVM), actually identifies more planets and overlooks fewer of them. As previously discussed, while the primary objective is to minimize missed planetary candidates, we must also avoid the pitfall of classifying every signal as a planet as observed with the Decision Tree model. False positives can typically be ruled out upon closer inspection by a trained eye. In this respect, the Support Vector Machine, with its superior Recall score, may be more advantageous for ensuring that potential planetary discoveries are not missed.

Alternatively, we could create a Cascade Model by feeding the data flagged as 'planet' by the SVM into the Random Forest model, which possesses higher Precision. Another option would be to construct a new Ensemble Model combining these two, utilizing the Soft Voting technique to refine the predictions.


<table>
  <tr>
    <td><img src="results\confusion_matrix_random_forest.png" alt="1" style="width: 500px"></td>
    <td><img src="results\confusion_matrix_svm.png" alt="2" style="width: 500px"></td>
    
  </tr>
</table>
 

# 4. Cascade Classifier Model
In the Cascade model configuration, the positive predictions generated by the Support Vector Machine (which possessed a higher Recall score of 0.902) were fed into the Random Forest model (characterized by a higher Precision score of 0.630). Following the modeling process, the resulting Cascade Classifier Model exhibited a decrease in Recall; however, its Precision score surpassed the individual scores of both constituent models. Consequently, the F1 score increased from 66.9% (Random Forest) to 67.2% (an increase of %0.3).


```python
cascade_result = pd.read_csv("cascade_metrics_report_shape.csv")
cascade_result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Best Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cascade Classifier (SVM+RF)</td>
      <td>-</td>
      <td>0.649</td>
      <td>0.697</td>
      <td>0.672</td>
    </tr>
  </tbody>
</table>
</div>



### 4.1. Confusion Matrix
Although the Cascade model achieved a higher F1 score, it falls behind both of the previously successful models regarding our primary objective of maximizing exoplanet discovery.<br>


<img src="results/cascade_confusion_matrix.png" alt="2" style="width: 500px">

# 5. Soft Voting Ensemble Model

Another approach to combine the performance of the two most successful models is the Ensemble Method. <br> 
In this method, we aggregate the two models and enable them to reach a consensus using a technique called Soft Voting. <br> 
The Ensemble model operates using default decision thresholds. The predicted probabilities for Class 1 (exoplanet) are retrieved from both models. These probabilities are summed and divided by two. The resulting average becomes the new predicted probability for the Ensemble model. <br> 
If this probability exceeds the Decision Threshold, the model generates a positive prediction. Following this calculation, we applied the decision threshold optimization algorithm to the Ensemble model and identified the new optimal decision threshold as 0.46.



```python
ensemble_result = pd.read_csv("ensemble_metrics_report.csv")
ensemble_result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Best Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Voting Ensemble (RF+SVM)</td>
      <td>0.46</td>
      <td>0.589</td>
      <td>0.787</td>
      <td>0.674</td>
    </tr>
  </tbody>
</table>
</div>



## 5.1. Ensemble Confusion Matrix
While the Ensemble Model detects a greater number of exoplanets (96) compared to the Random Forest model (87), it simultaneously mitigates the high false positive rate associated with the Support Vector Machine. <br> 
In this way, we successfully combined strengths of both models. 

<img src="results/ensemble_confusion_matrix.png" alt="2" style="width: 500px">

# 6. Final Results


```python
rf_svm = results[results["Model"].isin(["Random Forest", "SVM"])]
rf_svm = rf_svm[["Model", "Best Threshold", "Precision", "Recall", "New F1"]]
rf_svm = rf_svm.rename(columns={"New F1": "F1 Score"})
final_result = [rf_svm, cascade_result, ensemble_result]
final_df = pd.concat(final_result, ignore_index=True)
final_df["Best Threshold"] = pd.to_numeric(final_df["Best Threshold"], errors='coerce')
final_df["Best Threshold"] = final_df["Best Threshold"].round(2)
final_df = final_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
final_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Best Threshold</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Voting Ensemble (RF+SVM)</td>
      <td>0.46</td>
      <td>0.589</td>
      <td>0.787</td>
      <td>0.674</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cascade Classifier (SVM+RF)</td>
      <td>NaN</td>
      <td>0.649</td>
      <td>0.697</td>
      <td>0.672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest</td>
      <td>0.48</td>
      <td>0.630</td>
      <td>0.713</td>
      <td>0.669</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVM</td>
      <td>0.40</td>
      <td>0.516</td>
      <td>0.902</td>
      <td>0.657</td>
    </tr>
  </tbody>
</table>
</div>



# 7. Future Work 
A potential subsequent phase would be to feed this data into Convolutional Neural Networks (CNNs). <br>
Unlike the Random Forest algorithm, which relies on pre-calculated metrics (such as depth and skewness), a 1D-CNN can directly process raw and folded photometric time-series data. This capability allows the model to learn morphological features of planetary transits that might otherwise be lost during the binning process. <br>
However, effectively training such a network while avoiding overfitting would require a significantly larger dataset.


