import joblib
b = joblib.load('app/ml_models/severity/severity_triage_model_notebook.joblib')
print(list(b.keys()))
print(b.get('raw_inputs'))
print(b.get('completeness_floor'))
print(b.get('high_cut'))
print(b.get('low_cut'))
