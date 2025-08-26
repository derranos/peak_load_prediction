print("This is a test file for data processing.")
from data_processing.data_processing import load_data, process_data, target_distribution_by_year
from model_funcs.model_funcs import create_lightgbm_model, model_testing, model_accuracy, predict_month
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
historical_data = load_data("data/data_2022-2024.csv", start_date="2022-01-01")  # сырые данные
test_raw = load_data("data/data_2025.csv", start_date="2025-01-01")
full_data = pd.concat([historical_data, test_raw], ignore_index=True)
full_processed = process_data(full_data)
cutoff = pd.Timestamp("2025-01-01")
train_processed = full_processed[full_data["date"] < cutoff].reset_index(drop=True)
test_processed  = full_processed[full_data["date"] >= cutoff].reset_index(drop=True)
#target_distribution_by_year(load_data("data/data_2022-2024.csv", start_date="2022-01-01"))
#target_distribution_by_year(load_data("data/data_2025.csv", start_date="2025-01-01"))
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

real_2025 = load_data("data/data_2025.csv", start_date="2025-01-01")
real_2025 = real_2025.dropna(subset=["hour"])  # Удаляем NaN
real_2025['date'] = pd.to_datetime(real_2025['date'])


comparison = pd.merge(
    real_2025[['date', 'hour']],
    predictions[['date', 'predicted_hour']],
    on='date',
    how='inner'
)


if not comparison.empty:
    acc = accuracy_score(comparison['hour'], comparison['predicted_hour'])
    print(f"Accuracy предсказаний 2025: {acc:.3f}")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(comparison['hour'], comparison['predicted_hour'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation=90, cmap="Blues")
    plt.title("Confusion Matrix for 2025 Predictions")
    plt.show()
else:
    print("Нет данных для сравнения за 2025!")