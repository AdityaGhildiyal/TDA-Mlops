# Show the model.py content we will write to tda_detect/model.py
# (actual file creation done outside notebook — see model.py artifact)

model_py_preview = '''
tda_detect/model.py
===================
TDAAnomalyDetector — wraps TDAFeatureExtractor + IsolationForest
into a single sklearn-compatible estimator.

Public API
----------
    detector = TDAAnomalyDetector()
    detector.fit(raw_signals)          # list of 1-D arrays, normal only
    scores   = detector.score(signals) # decision_function output
    labels   = detector.predict(signals, threshold)
    f1       = detector.evaluate(signals, y_true, threshold)
'''
print(model_py_preview)

# Persist the trained classifier and threshold for Phase 4
import pickle, pathlib

pathlib.Path("models").mkdir(exist_ok=True)
with open("models/isoforest_phase3.pkl", "wb") as f:
    pickle.dump({"clf": clf, "threshold": best_thr,
                 "contamination": "auto"}, f)
print("Saved: models/isoforest_phase3.pkl")
print(f"  clf            : {clf}")
print(f"  best_threshold : {best_thr:.6f}")