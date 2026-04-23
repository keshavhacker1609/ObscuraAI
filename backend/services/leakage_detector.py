import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def detect_leakage():

    # SAMPLE DATA (replace later with real CSV)
    data = pd.DataFrame({
        "feature1":[1,2,3,4,5,6,7,8],
        "feature2":[5,4,3,2,1,2,3,4],
        "sensitive":[0,1,0,1,0,1,0,1],
        "target":[1,0,1,0,1,0,1,0]
    })

    X = data[["feature1","feature2"]]
    y = data["target"]
    sensitive = data["sensitive"]

    # MAIN MODEL
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ATTACKER MODEL (detect leakage)
    attacker = RandomForestClassifier()
    attacker.fit(X_test, sensitive.iloc[:len(X_test)])

    attacker_pred = attacker.predict(X_test)

    leakage_score = accuracy_score(
        sensitive.iloc[:len(X_test)],
        attacker_pred
    )

    return {
        "leakage_score": float(leakage_score),
        "risk_level": "HIGH" if leakage_score > 0.7 else "LOW"
    }
