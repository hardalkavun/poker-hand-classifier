import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import time

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. VERİ YÜKLEME VE ÖN İŞLEME
def load_and_preprocess_data():
    print("Veri seti yükleniyor...")
    poker_hand = fetch_ucirepo(id=158)
    X = poker_hand.data.features
    y = poker_hand.data.targets.values.ravel()

    print("\nSınıf Dağılımı:")
    print(pd.Series(y).value_counts(normalize=True))

    return X, y

# 2. ÖZELLİK MÜHENDİSLİĞİ
def feature_engineering(X):
    print("\nÖzellik mühendisliği uygulanıyor...")
    df = X.copy()

    rank_cols = ['C1', 'C2', 'C3', 'C4', 'C5']
    suit_cols = ['S1', 'S2', 'S3', 'S4', 'S5']

    for col in rank_cols:
        df[col] = df[col].replace({14: 1})

    df['is_flush'] = (df[suit_cols].nunique(axis=1) == 1).astype(int)

    def is_straight(ranks):
        ranks = sorted(set(ranks))
        if len(ranks) != 5:
            return 0
        return int(ranks[-1] - ranks[0] == 4)

    df['is_straight'] = df[rank_cols].apply(is_straight, axis=1)

    def hand_type(ranks):
        counts = pd.Series(ranks).value_counts()
        if 4 in counts.values:
            return 'four_of_a_kind'
        elif 3 in counts.values and 2 in counts.values:
            return 'full_house'
        elif 3 in counts.values:
            return 'three_of_a_kind'
        elif (counts == 2).sum() == 2:
            return 'two_pair'
        elif (counts == 2).sum() == 1:
            return 'one_pair'
        else:
            return 'high_card'

    df['hand_type'] = df[rank_cols].apply(hand_type, axis=1)

    hand_type_map = {
        'four_of_a_kind': 6,
        'full_house': 5,
        'three_of_a_kind': 4,
        'two_pair': 3,
        'one_pair': 2,
        'high_card': 1
    }
    df['hand_type'] = df['hand_type'].map(hand_type_map)

    df['high_card'] = df[rank_cols].max(axis=1)

    return df

# 3. MODEL EĞİTİMİ (SMOTE + Random Forest)
def train_model(X_train, y_train):
    print("\nModel eğitimi başlatılıyor...")

    print("SMOTE ile sınıf dengeleniyor...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=42
    )

    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring='f1_macro',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    search.fit(X_train_resampled, y_train_resampled)
    training_time = time.time() - start_time

    print(f"\nEğitim süresi: {training_time:.2f} saniye")
    print("🎯 En iyi parametreler:", search.best_params_)
    print("🏆 En iyi F1 skoru:", search.best_score_)

    return search.best_estimator_

# 4. DEĞERLENDİRME
def evaluate_model(model, X_test, y_test):
    print("\nModel değerlendiriliyor...")
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    adjusted_report = {}

    for label, metrics in report.items():
        if isinstance(metrics, dict):
            adjusted_report[label] = {
                metric: (round(value, 4) if value > 0 else 0.0001)
                for metric, value in metrics.items()
            }
        else:
            adjusted_report[label] = metrics

    print("\nClassification Report (sıfırlar yerine küçük değerler ile):")
    print(pd.DataFrame(adjusted_report).T)

    f1_macro = report['macro avg']['f1-score']
    f1_weighted = report['weighted avg']['f1-score']

    print(f"\nMacro F1 Skoru: {f1_macro:.4f}")
    print(f"Weighted F1 Skoru: {f1_weighted:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    print("\nÖzellik Önem Grafiği:")
    importances = model.feature_importances_
    feature_names = [str(i) for i in range(X_test.shape[1])]
    if hasattr(X_test, 'columns'):
        feature_names = X_test.columns

    plt.figure(figsize=(10,6))
    plt.barh(feature_names, importances)
    plt.xlabel("Önem Derecesi")
    plt.title("Özellik Önem Grafiği")
    plt.show()

# 5. MODEL KAYDETME
def save_model(model, filename='best_random_forest_model.pkl'):
    joblib.dump(model, filename)
    print(f"\nModel '{filename}' olarak kaydedildi.")

# ANA İŞLEM
def main():
    try:
        X, y = load_and_preprocess_data()
        X = feature_engineering(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        best_model = train_model(X_train, y_train)
        evaluate_model(best_model, X_test, y_test)
        save_model(best_model)

    except Exception as e:
        print(f"\n Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()