# Gerekli kütüphanelerin import edilmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import time

# Uyarıları görmezden gel
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. VERİ YÜKLEME VE ÖN İŞLEME
def load_and_preprocess_data():
    print("Veri seti yükleniyor")
    poker_hand = fetch_ucirepo(id=158)  # UCI'dan Poker Hand veri setini indir
    X = poker_hand.data.features         # Özellikler
    y = poker_hand.data.targets.values.ravel()  # Hedef değişkeni (sınıf etiketi)

    # Sınıf dağılımını yüzde olarak yazdır
    print("\nSınıf Dağılımı:")
    print(pd.Series(y).value_counts(normalize=True))

    return X, y

# 2. ÖZELLİK MÜHENDİSLİĞİ
def feature_engineering(X):
    print("\nÖzellik mühendisliği uygulanıyor...")
    df = X.copy()  # Orijinal veriyi bozmadan kopya al

    # Rank (değer) ve Suit (renk) sütunlarını ayır
    rank_cols = ['C1', 'C2', 'C3', 'C4', 'C5']
    suit_cols = ['S1', 'S2', 'S3', 'S4', 'S5']

    # Eğer 14 varsa (As), 1 olarak değiştir
    for col in rank_cols:
        df[col] = df[col].replace({14: 1})

    # Bütün kartlar aynı renkteyse flush olduğunu belirten yeni bir özellik
    df['is_flush'] = (df[suit_cols].nunique(axis=1) == 1).astype(int)

    # Straight olup olmadığını belirleyen yardımcı fonksiyon
    def is_straight(ranks):
        ranks = sorted(set(ranks))
        if len(ranks) != 5:
            return 0
        return int(ranks[-1] - ranks[0] == 4)

    # Straight kontrolü
    df['is_straight'] = df[rank_cols].apply(is_straight, axis=1)

    # El tipini belirleyen yardımcı fonksiyon
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

    # Her satır için el tipini belirle
    df['hand_type'] = df[rank_cols].apply(hand_type, axis=1)

    # El tipi kategorilerini sayısal değerlere çevir
    hand_type_map = {
        'four_of_a_kind': 6,
        'full_house': 5,
        'three_of_a_kind': 4,
        'two_pair': 3,
        'one_pair': 2,
        'high_card': 1
    }
    df['hand_type'] = df['hand_type'].map(hand_type_map)

    # En yüksek kartı belirten özellik
    df['high_card'] = df[rank_cols].max(axis=1)

    return df

# 3. MODEL EĞİTİMİ (SMOTE + Naive Bayes)
def train_model(X_train, y_train):
    print("\nModel eğitimi başlatılıyor...")

    # SMOTE ile veri dengeleniyor (azınlık sınıflar çoğaltılıyor)
    print("SMOTE ile sınıf dengeleniyor...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = GaussianNB()  # Naive Bayes modeli

    # Modelin eğitim süresini ölç
    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    training_time = time.time() - start_time

    print(f"\nEğitim süresi: {training_time:.2f} saniye")

    return model

# 4. DEĞERLENDİRME
def evaluate_model(model, X_test, y_test):
    print("\nModel değerlendiriliyor...")
    y_pred = model.predict(X_test)  # Test verisiyle tahmin

    # Sınıflandırma raporunu hesapla
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    adjusted_report = {}

    # Raporu okunabilir hale getir, sıfır değerler yerine çok küçük değerler yaz
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

    # F1 skorlarını yazdır
    f1_macro = report['macro avg']['f1-score']
    f1_weighted = report['weighted avg']['f1-score']

    print(f"\nMacro F1 Skoru: {f1_macro:.4f}")
    print(f"Weighted F1 Skoru: {f1_weighted:.4f}")

    # Karışıklık matrisi oluştur ve görselleştir
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')  #mavi
    plt.title("Confusion Matrix")
    plt.show()

# 5. MODEL KAYDETME
def save_model(model, filename='best_naive_bayes_model.pkl'):
    joblib.dump(model, filename)  # Modeli diske kaydet
    print(f"\nModel '{filename}' olarak kaydedildi.")

# ANA İŞLEM
def main():
    try:
        # Veri yükleme ve işleme
        X, y = load_and_preprocess_data()
        X = feature_engineering(X)


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y  #veri eğitim seti ve test seti olarak ayır
        )

        # Özellikleri ölçeklendir (standartlaştır)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Modeli eğit, değerlendir ve kaydet
        best_model = train_model(X_train, y_train)
        evaluate_model(best_model, X_test, y_test)
        save_model(best_model)

    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")

#model çalıştır ??
if __name__ == "__main__":
    main()
