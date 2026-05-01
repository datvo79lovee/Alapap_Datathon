# DATATHON 2026 - Revenue & COGS Forecasting

## Tong quan

Du an du bao doanh thu (Revenue) va chi phi hang ban (COGS) cho 548 ngay (2023-01-01 den 2024-07-01).

**Mo hinh:** Logistic Regression (Classification)
**Cross-Validation:** TimeSeriesSplit (5 folds)
**CV Accuracy:** 79.67%

---

## Cau lenh chay

```bash
python v10_model.py
```

---

## Thu vien can cai dat

```bash
pip install pandas numpy scikit-learn
```

---

## Cau truc thu muc

```
Alapap_Datathon/
├── v10_model.py              # Code chinh (chay de tao ket qua)
├── final.csv                # Ket qua du bao (548 ngay)
├── metrics.csv               # Cac chi so hieu suat (MAE, RMSE, MAPE, R2)
├── feature_importance.csv   # Phan tich Feature Importance
├── sales.csv                # Du lieu train (2012-2022)
├── sample_submission.csv    # Mau submission
├── EDA.ipynb                # Notebook phan tich du lieu
├── MCQ.ipynb                # Notebook cau hoi trac nghiem
├── REPORT.md                # Bao cao ky thuat
└── README.md               # File nay
```

---

## Huong dan cho nguoi cham bai

### Cach 1: Kiem tra ket qua (khong can chay lai code)

1. **Xem ket qua du bao:**
   ```bash
   # Mo file final.csv
   # 548 dong: Date, Revenue, COGS
   ```

2. **Xem chi so hieu suat:**
   ```bash
   # Mo file metrics.csv
   # Co: MAE, RMSE, MAPE, R2
   ```

3. **Xem phan tich Feature Importance:**
   ```bash
   # Mo file feature_importance.csv
   ```

### Cach 2: Chay lai code (tai lap ket qua)

1. **Cai dat thu vien:**
   ```bash
   pip install pandas numpy scikit-learn
   ```

2. **Chay code:**
   ```bash
   python v10_model.py
   ```

3. **Ket qua se tao:**
   - `final.csv` - Ket qua du bao
   - `metrics.csv` - Cac chi so
   - `feature_importance.csv` - Phan tich

---

## Chi so hieu suat (tu file metrics.csv)

| Chỉ số | Giá trị | Mo ta |
|--------|---------|-------|
| MAE | 1,132,335 | Sai so tuyet doi trung binh |
| RMSE | 1,506,362 | Can bac 2 cua binh phuong sai so |
| MAPE | 37.98% | Phan tram sai so tuyet doi trung binh |
| R² | 0.0913 | He so xac dinh |

**Chu y:** Cac chi so nay duoc tinh so voi sample_submission. Khi Kaggle cham diem, se tinh tren actual test data.

---

## Feature Importance (Top 5)

| Feature | Coefficient | Y nghia |
|---------|-------------|---------|
| is_month_end | +1.128 | Cuoi thang doanh thu cao hon |
| quarter | -0.655 | Quy anh huong tiêu cực |
| month_cos | -0.574 | Chu ky thang |
| is_weekend | -0.397 | Cuoi tuan doanh thu thap hon |
| month_sin | -0.356 | Chu ky thang |

---

## Ràng buoc (Constraints)

| Ràng buộc | Status |
|-----------|--------|
| (1) Khong dung Revenue/COGS test lam features | ✅ |
| (2) Khong dung du lieu ngoai | ✅ |
| (3) Co ma nguon tai lap | ✅ |
| (4) Cross-validation dung thu tu thoi gian | ✅ |

---

## Lien he

**Link GitHub:** https://github.com/datvo79lovee/Alapap_Datathon

---

**Ghi chu:** 
- Mo hinh su dung Logistic Regression (Classification) de du doan xac suat "ngay doanh thu cao", sau do chuyen doi thanh Revenue.
- COGS duoc tinh theo ty le 87.23% tu du lieu 2022.