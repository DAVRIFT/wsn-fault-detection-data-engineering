from imblearn.under_sampling import RandomUnderSampler

def apply_rus(X, y):
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X, y)
