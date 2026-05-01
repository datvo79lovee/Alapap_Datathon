# BAO CAO KY THUAT - DATATHON 2026
## Revenue & COGS Forecasting

---

## 1. MO TA BAI TOAN

- **Input**: Dữ liệu bán hàng từ 2012-2022 (3,833 ngày)
- **Output**: Dự báo Revenue và COGS cho 548 ngày (2023-01-01 đến 2024-07-01)
- **Metric**: MAPE (Mean Absolute Percentage Error)

---

## 2. EDA - PHAN TICH DU LIEU

### 2.1 Tong quan du lieu
- Tong so ngay: 3,833 (2012-07-04 den 2022-12-31)
- Cac cot: Date, Revenue, COGS

### 2.2 Xu huong theo nam
| Nam | Revenue (B) | YoY |
|-----|------------|-----|
| 2012 | 0.74 | - |
| 2013 | 1.66 | +123.5% |
| 2014 | 1.87 | +13.0% |
| 2015 | 1.89 | +1.0% |
| 2016 | 2.10 | +11.4% (peak) |
| 2017 | 1.91 | -9.2% |
| 2018 | 1.85 | -3.2% |
| 2019 | 1.14 | -38.6% (COVID) |
| 2020 | 1.05 | -7.2% |
| 2021 | 1.04 | -1.1% |
| 2022 | 1.17 | +12.1% (recovery) |

### 2.3 Pattern theo thang
- Thang cao nhat: April, May, June (4.5-4.6M)
- Thang thap nhat: January, December (1.8M)

---

## 3. FEATURE ENGINEERING

### 3.1 Cac dac trung su dung (12 features)

| Loai | Features |
|------|----------|
| Time | week, quarter |
| Binary | is_weekend, is_month_start, is_month_end |
| Cyclical | month_sin, month_cos |
| Lag | lag_7, lag_14, lag_21 |
| Rolling | rma_7, rma_14 |

### 3.2 Bien muc tieu
- **is_high**: Phan loai ngay doanh thu cao (Revenue > monthly median)
- Phan bo: 50% high, 50% low

---

## 4. MO HINH

### 4.1 Chon Logistic Regression (Classification)
- Ly do: Don gian, nhanh, de giai thich
- Output: Xac suat ngay "doanh thu cao"
- Chuyen doi: Revenue = mean_2022 * (0.8 + 0.4 * probability)

### 4.2 Training
- Du lieu: 2022 (365 ngay)
- Target: is_high (binary)

---

## 5. CROSS-VALIDATION

### 5.1 Phuong phap: TimeSeriesSplit (5 folds)
- Dung thu tu thoi gian, khong shuffle
- Dam bao khong co data leakage

### 5.2 Ket qua
| Fold | Accuracy |
|------|----------|
| 1 | 71.67% |
| 2 | 76.67% |
| 3 | 75.00% |
| 4 | 88.33% |
| 5 | 86.67% |
| **Mean** | **79.67% (±6.62%)** |

---

## 6. FEATURE IMPORTANCE

### 6.1 Coefficient cua Logistic Regression

| Feature | Coefficient | Y nghia |
|---------|-------------|---------|
| is_month_end | +1.128 | Cuoi thang doanh thu cao hon |
| quarter | -0.655 | Quy anh huong tiêu cực |
| month_cos | -0.574 | Chu ky thang |
| is_weekend | -0.397 | Cuoi tuan doanh thu thap hon |
| month_sin | -0.356 | Chu ky thang |

### 6.2 Giai thich
- **is_month_end** = 1.128: Ngay cuoi thang (26-31) co xac suat doanh thu cao cao hon
- **is_weekend** = -0.397: Cuoi tuan (Sat-Sun) co xac suat doanh thu cao thap hon
- Cac feature lag/rolling co he so nho, khong anh huong nhieu

---

## 7. KET QUA DU BAO

### 7.1 Thong ke
- Ngay: 548 (2023-01-01 den 2024-07-01)
- Revenue Mean: 2,883,794
- COGS Mean: 2,515,652
- COGS Ratio: 87.23%

### 7.2 So sanh voi sample
- Sample mean: 3,249,795
- Our mean: 2,883,794
- Chenh lech: -11.26%

---

## 8. RANG BUOC

| Ràng buộc | Status |
|-----------|--------|
| (1) Khong dung Revenue/COGS test lam features | ✅ |
| (2) Khong dung du lieu ngoai | ✅ |
| (3) Co ma nguon tai lap | ✅ |
| (4) Cross-validation dung thu tu thoi gian | ✅ |

---

## 9. KET LUAN

- Mo hinh Logistic Regression dat 79.67% accuracy trong cross-validation
- Feature quan trong nhat: is_month_end, quarter, is_weekend
- COGS ratio su dung: 87.23% (tu 2022 data)

---

**Link Code**: https://github.com/datvo79lovee/Alapap_Datathon