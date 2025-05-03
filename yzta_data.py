# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping

# 🔹 Veriyi yükle
df = pd.read_csv("/kaggle/input/yzta-datathon/train.csv")
test = pd.read_csv("/kaggle/input/yzta-datathon/testFeatures.csv")

# 🔹 Tarih sütununu yıl ve aya ayır
for d in [df, test]:
    d["tarih"] = pd.to_datetime(d["tarih"])
    d["yıl"] = d["tarih"].dt.year
    d["ay"] = d["tarih"].dt.month
    d.drop("tarih", axis=1, inplace=True)

# 🔹 Özellik mühendisliği - TRAIN
df["ürün_market"] = df["ürün"].astype(str) + "_" + df["market"].astype(str)
ürün_fiyat_ort = df.groupby("ürün")["ürün fiyatı"].mean().to_dict()
df["ürün_ortalama_fiyat"] = df["ürün"].map(ürün_fiyat_ort)
şehir_ürün_ort = df.groupby(["şehir", "ürün"])["ürün fiyatı"].mean().to_dict()
df["şehir_ürün_ort"] = df.set_index(["şehir", "ürün"]).index.map(şehir_ürün_ort)
ay_fiyat_ort = df.groupby("ay")["ürün fiyatı"].mean().to_dict()
df["ay_ortalama_fiyat"] = df["ay"].map(ay_fiyat_ort)

# 🔹 Aynı dönüşümleri TEST'e uygula
test["ürün_market"] = test["ürün"].astype(str) + "_" + test["market"].astype(str)
test["ürün_ortalama_fiyat"] = test["ürün"].map(ürün_fiyat_ort)
test["şehir_ürün_ort"] = test.set_index(["şehir", "ürün"]).index.map(şehir_ürün_ort)
test["ay_ortalama_fiyat"] = test["ay"].map(ay_fiyat_ort)

# 🔹 Kategorik sütunları tanımla ve encode et
cat_cols = ["ürün", "ürün kategorisi", "ürün üretim yeri", "market", "şehir", "ürün_market"]
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([df[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    df[col] = le.transform(df[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 🔹 Özellik ve hedef değişken
X = df.drop("ürün fiyatı", axis=1)
y = df["ürün fiyatı"]

# 🔹 Eğitim - doğrulama ayrımı
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 LightGBM Modeli
model = LGBMRegressor(
    objective='regression',
    metric='rmse',
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=1000
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[early_stopping(stopping_rounds=50)],
    categorical_feature=cat_cols
)

# 🔹 Doğrulama sonucu
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print("✅ Yeni Doğrulama RMSE:", rmse)

# 🔹 Test seti için tahmin ve submission
ids = test["id"]
X_test = test.drop("id", axis=1)
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "id": ids,
    "ürün fiyatı": predictions
})
submission.to_csv("submission.csv", index=False)
print("📁 Yeni 'submission.csv' başarıyla oluşturuldu.")
