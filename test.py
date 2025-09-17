print("This is a test file for data processing.")
from data_processing.data_processing import load_data, process_data, target_distribution_by_year, target_distribution_by_month, target_distribution_by_dayofweek
from model_funcs.model_funcs import create_lightgbm_model, model_testing, model_accuracy, predict_month, plot_confusion_matrix_and_metrics
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
historical_data = load_data("data/data_2022-2024.csv", start_date="2022-01-01")  # сырые данные
test_raw = load_data("data/data_2025.csv", start_date="2025-01-01")
full_data = pd.concat([historical_data, test_raw], ignore_index=True)
full_processed = process_data(full_data)
cutoff = pd.Timestamp("2025-01-01")
train_processed = full_processed[full_data["date"] < cutoff].reset_index(drop=True)
test_processed  = full_processed[full_data["date"] >= cutoff].reset_index(drop=True)

# target_distribution_by_year(full_data)
# target_distribution_by_month(full_data[full_data["date"].dt.year == 2022])
# target_distribution_by_month(full_data[full_data["date"].dt.year == 2023])
# target_distribution_by_month(full_data[full_data["date"].dt.year == 2024]) 
# target_distribution_by_month(full_data[full_data["date"].dt.year == 2025])
# target_distribution_by_dayofweek(full_data[full_data["date"].dt.year == 2022])
# target_distribution_by_dayofweek(full_data[full_data["date"].dt.year == 2023])
# target_distribution_by_dayofweek(full_data[full_data["date"].dt.year == 2024])
# target_distribution_by_dayofweek(full_data[full_data["date"].dt.year == 2025])

data_for_clustering = full_processed[["dow", "month", "is_weekend", "hour"]].values
db = DBSCAN(eps=1.5, min_samples=15).fit(data_for_clustering)
labels = db.labels_
# Пометка аномалий (-1 означает шум)
full_processed['is_anomaly'] = labels
anomalies = full_processed[full_processed['is_anomaly'] == -1]
print(f"Найдено аномалий: {len(anomalies)}")
print("Примеры аномалий:\n", anomalies.head())

a = input("Press Enter to continue...")
#target_distribution_by_year(process_data(load_data("data/data_2025.csv", start_date="2025-01-01")))
model = create_lightgbm_model()
model, acc1 = model_testing(model, train_processed)
joblib.dump(model, "model_funcs/lightgbm_model.pkl")

acc2 = model_accuracy(model, test_processed)
print(f"Model accuracy on 2025 data: {acc2:.3f}")
print(f"Model accuracy during training: {acc1:.3f}")
predictions = pd.DataFrame()
for i in range(1,8):
    month_pred = predict_month(model, month=i, year=2025)
    if month_pred is not None:
        predictions = pd.concat([predictions, month_pred], ignore_index=True)

plot_confusion_matrix_and_metrics(model,None,test_raw["hour"],predictions["predicted_hour"], title="Confusion Matrix for 2025 Predictions")