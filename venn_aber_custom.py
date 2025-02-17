import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from venn_abers import VennAbersCalibrator

# create sample dataset
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=73)
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=73)

# Fit model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model predict calibration set
p_cal = model.predict_proba(X_train)

cal_scores = p_cal[:, 1]

def venn_abers_predictor(cal_scores, cal_labels, test_scores):
    p0_list, p1_list = [], []
    
    for test_score in test_scores:
        s0 = np.append(cal_scores, test_score)
        s1 = np.append(cal_scores, test_score)
        
        # Compute g0
        iso_reg_0 = IsotonicRegression(out_of_bounds='clip', increasing='auto')
        g0 = iso_reg_0.fit_transform(s0, np.append(cal_labels, 0))
        p0 = g0[-1]
        
        # Compute g1
        iso_reg_1 = IsotonicRegression(out_of_bounds='clip', increasing='auto')
        g1 = iso_reg_1.fit_transform(s1, np.append(cal_labels, 1))
        p1 = g1[-1]
        
        p0_list.append(p0)
        p1_list.append(p1)
    
    return np.array(p0_list), np.array(p1_list)


p_test = model.predict_proba(X_cal)
test_scores = p_test[:, 1]  


p0_test_calibrated, p1_test_calibrated = venn_abers_predictor(cal_scores, y_train, test_scores)

p_final = p1_test_calibrated / (1 - p0_test_calibrated + p1_test_calibrated)

n = 10
print("Venn-Abers implement:")
print("p0:", p0_test_calibrated[0:n])
print("p1:", p1_test_calibrated[0:n])
print("Final calibrated probabilities:", p_final[0:n])

va = VennAbersCalibrator()
pred_calibrated, p0p1 = va.predict_proba(p_cal=p_cal, y_cal=y_train, p_test=p_test, p0_p1_output = True)
print(p0p1[:n])