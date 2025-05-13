import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import joblib
import time


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
            return 6
        elif 3 in counts.values and 2 in counts.values:
            return 5
        elif 3 in counts.values:
            return 4
        elif (counts == 2).sum() == 2:
            return 3
        elif (counts == 2).sum() == 1:
            return 2
        else:
            return 1

    df['hand_type'] = df[rank_cols].apply(hand_type, axis=1)
    df['high_card'] = df[rank_cols].max(axis=1)

    return df


# 3. MODEL EĞİTİMİ (SMOTE + Yapay Sinir Ağı)
def train_ann(X_train, y_train, X_test, y_test):
    print("\nModel eğitimi başlatılıyor...")

    print("SMOTE ile sınıf dengeleniyor...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    y_train_resampled = tf.keras.utils.to_categorical(y_train_resampled, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_resampled.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    start_time = time.time()
    history = model.fit(X_train_resampled, y_train_resampled,
                        validation_data=(X_test, y_test),
                        batch_size=32, epochs=100,
                        callbacks=[early_stopping], verbose=2)

    training_time = time.time() - start_time

    print(f"\nEğitim süresi: {training_time:.2f} saniye")

    return model, history


# 4. DEĞERLENDİRME
def evaluate_model(model, X_test, y_test):
    print("\n🔍 Model Değerlendirme Başlatılıyor...")

    y_pred_classes = np.argmax(model.predict(X_test), axis=1)
    y_test_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

    cm = confusion_matrix(y_test_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    print("\n🔎 Sınıf Bazlı Değerlendirme:")
    report = classification_report(y_test_classes, y_pred_classes, digits=4)
    print(report)


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Doğruluk Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu', color='blue', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validasyon Doğruluğu', color='red', marker='s')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Doğruluk")
    plt.grid(True)

    # Kayıp Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı', color='blue', marker='o')
    plt.plot(history.history['val_loss'], label='Validasyon Kaybı', color='red', marker='s')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp")
    plt.grid(True)

    plt.show()


# 5. MODEL KAYDETME
def save_model(model, filename='ann_poker_hand_model.h5'):
    model.save(filename)
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

        ann_model, history = train_ann(X_train, y_train, X_test, y_test)
        save_model(ann_model)
        plot_training_history(history)

        evaluate_model(ann_model, X_test, y_test)

        print("\n✅ Model eğitimi ve değerlendirmesi tamamlandı!")

    except Exception as e:
        print(f"\n❌ Hata oluştu: {str(e)}")


if __name__ == "__main__":
    main()