import time
import lightkurve as lk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from lightkurve import TessLightCurve

from lightkurve.lightcurve import TessLightCurve
from lightkurve.periodogram import BoxLeastSquaresPeriodogram, LombScarglePeriodogram

import warnings
warnings.filterwarnings("ignore")


def clean_outliers(tid: str) -> TessLightCurve | None: 

    """
    Cleans the light curve by removing nans, removing outliers, and flattening the light curve.
    """

    try:

        search: lk.SearchResult = lk.search_lightcurve(f"TIC {tid}", mission="TESS", author="SPOC")
        if len(search) == 0:
            return None
        
        
        lc: TessLightCurve = search[0].download()


        lc = lc.remove_nans()

        lc = lc[lc.quality == 0]

        lc = lc.remove_outliers(sigma = 3)

        flat_lc = lc.flatten(window_length = 401)

        return flat_lc

    except Exception as e:
        print(f"Error processing TIC {tid}: {e}")
        return None
    


def feature_extraction_with_shape(lc: TessLightCurve, n_bins=50) -> dict:
    """
    Extracts scalar features AND the binned folded lightcurve shape.
    Returns a dictionary suitable for appending to a DataFrame.
    """
    features = {}

    # statistical
    

    # physical (BLS)
    try:
        # BLS Periodogram
        
        pg = lc.to_periodogram(method="bls", period=np.linspace(1, 15, 5000))
        
        best_period = pg.period_at_max_power.value
        best_t0 = pg.transit_time_at_max_power.value
        
        features['period'] = best_period
        features['transit_depth'] = pg.depth_at_max_power.value
        features['transit_duration'] = pg.duration_at_max_power.value
        features['max_power'] = pg.max_power.value

        # 2. Fold the Lightcurve
        # "Fold" means stacking all transits on top of each other based on the Period
        folded_lc = lc.fold(period=best_period, epoch_time=best_t0)

        # Statistical features of the folded lightcurve
        try:
            features["std_dev"] = np.nanstd(folded_lc.flux.value)
            features["skewness"] = skew(folded_lc.flux.value, nan_policy='omit')
            features["kurtosis"] = kurtosis(folded_lc.flux.value, nan_policy='omit')
        except:
            features["std_dev"], features["skewness"], features["kurtosis"] = 0, 0, 0

        # 3. Bin the Folded Curve
        # Converts thousands of points into a fixed size vector (e.g., 50 points)
        binned_lc = folded_lc.bin(bins=n_bins)
        
        # Normalize the binned flux (so 1.0 is baseline)
        # We fill NaNs with 1.0 (baseline) just in case a bin is empty
        flux_vector = np.nan_to_num(binned_lc.flux.value, nan=1.0)
        
        # 4. Add Shape Features to Dictionary
        # Create columns like 'flux_0', 'flux_1', ... 'flux_49'
        for i in range(n_bins):
            # Safety check: ensure we don't go out of bounds if binning behaves oddly
            val = flux_vector[i] if i < len(flux_vector) else 1.0
            features[f"flux_{i}"] = val

    except Exception as e:
        # Fallback if BLS fails (e.g., data is too noisy or flat)
        # print(f"BLS Error: {e}") # Optional: Uncomment to debug
        features['period'] = 0
        features['transit_depth'] = 0
        features['transit_duration'] = 0
        features['max_power'] = 0
        
        # Fill shape features with 1.0 (flat line)
        for i in range(n_bins):
            features[f"flux_{i}"] = 1.0

    
    return features

def batch_processing(df: pd.DataFrame) -> pd.DataFrame:
    data_matrix = []
    
    
    counter = 0
    failed = 0
    start_time = time.time()
    total_items = len(df)

    for index, row in df.iterrows():
        tic_id = row["tid"]
        label = row["target"]

        print(f"Processing TIC {tic_id}...")

        # Assuming clean_outliers downloads/loads the lightcurve
        lc = clean_outliers(tic_id)

        if lc:
            # --- CHANGE IS HERE ---
            # Use the new function with binning
            features = feature_extraction_with_shape(lc)
            
            # Add metadata manually
            features["target"] = label

            data_matrix.append(features)
            print("Done.")
            counter += 1
        else:
            print(f"Failed to load TIC {tic_id}")
            failed += 1

        # --- UNIFIED TIMER LOGIC ---
        # Run this block regardless of success/failure
        processed_count = counter + failed
        elapsed_time = time.time() - start_time
        
        if processed_count > 0:
            avg_time_per_item = elapsed_time / processed_count
            remaining_items = total_items - processed_count
            estimated_remaining_time = avg_time_per_item * remaining_items
            rem_min, rem_sec = divmod(estimated_remaining_time, 60)
            
            print(f"Progress: {processed_count}/{total_items} ({((processed_count/total_items)*100):.1f}%)")
            print(f"Est. Remaining: {rem_min:.0f}m {rem_sec:.0f}s")
            print("-" * 30)

    print(f"\nBatch Processing Complete.")
    print(f"Successfully processed: {counter}")
    print(f"Failed: {failed}")
    
    # Returns a DataFrame with ~58 columns (physics + flux bins + target + ID)
    

    return pd.DataFrame(data_matrix)



