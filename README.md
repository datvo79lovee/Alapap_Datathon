# DATATHON 2026 - Revenue & COGS Forecasting

## Mo ta
Du an du bao doanh thu va chi phi hang ban (COGS) cho giai doan 2023-01-01 den 2024-07-01 (548 ngay).

## Thanh phan

### Ma nguon
- `v10_model.py` - Model Logistic Regression voi feature importance va cross-validation

### Du lieu
- `sales.csv` - Du lieu ban hang 2012-2022
- `sample_submission.csv` - Mau submission

### Ket qua
- `final.csv` - Du bao cuoi cung (548 ngay)

## Cach chay

```bash
python v10_model.py
```

## Thu vien can cai
```bash
pip install pandas numpy scikit-learn
```

## Ket qua mo hinh
- Cross-validation Accuracy: 79.67%
- COGS ratio: 87.23%

## Giai thich model
Xem phan Feature Importance trong code (phan 5).

## Yeu cau
- Su dung Logistic Regression (Classification)
- Khong dung test data lam features
- Cross-validation theo thoi gian (TimeSeriesSplit)