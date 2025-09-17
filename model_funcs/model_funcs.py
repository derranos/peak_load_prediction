import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd
from data_processing.data_processing import process_data, load_data
def plot_confusion_matrix_and_metrics(model, X, y, yPred = None, title="Confusion Matrix"):
    if yPred is None:
        y_pred = model.predict(X)
    else:
        y_pred = yPred
    cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
    acc_cm = np.trace(cm) / np.sum(cm)
    print("Accuracy из матрицы:", acc_cm)
    f1 = f1_score(y, y_pred, average='weighted')
    print("F1 Score:", f1)
    recall = recall_score(y, y_pred, average='weighted')
    print("Recall:", recall)
    precision = precision_score(y, y_pred, average='weighted')
    print("Precision:", precision)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
    disp.plot(xticks_rotation=90, cmap="Blues")
    plt.title(title)
    plt.show()

def create_lightgbm_model():
    model = lgb.LGBMClassifier(
        objective="multiclass",   # Тип задачи: мультиклассовая классификация
        num_class=24,             # Количество классов (в твоём случае — 24 часа)
        boosting_type="gbdt",     # Алгоритм бустинга: Gradient Boosted Decision Trees
        learning_rate=0.1105,     # Шаг обучения: чем меньше, тем аккуратнее шаги и стабильнее модель,
                                  # но нужно больше деревьев (итераций)
        n_estimators=3000,        # Максимальное количество деревьев (итераций бустинга).
                                  # Итоговое число обычно меньше из-за early stopping.
        max_depth=-1,             # Максимальная глубина дерева. 
                                  # -1 = без ограничений. Большие значения → риск переобучения.
        num_leaves=27,            # Максимальное количество листьев в дереве.
                                  # Чем больше, тем сложнее дерево (больше правил), но выше риск переобучения.
        min_data_in_leaf=20,      # Минимальное число объектов в каждом листе.
                                  # Увеличение этого значения снижает переобучение, но может недообучать.
        feature_fraction=0.807,     # Доля признаков, выбираемых случайно для построения каждого дерева (аналог colsample_bytree).
                                  # Значение <1.0 добавляет случайности → меньше переобучение.
        bagging_fraction=0.8,     # Доля строк (объектов), используемых для каждого дерева (аналог subsample).
                                  # Тоже метод борьбы с переобучением.
        bagging_freq=1,           # Как часто применять bagging. 
                                  # 1 = применять при каждом построении дерева.
        lambda_l1=0.105,            # L1-регуляризация (штраф за абсолютные веса).
                                  # Сжимает коэффициенты → убирает слабые признаки.
        lambda_l2=0.105,            # L2-регуляризация (штраф за квадраты весов).
                                  # Сглаживает модель, снижает переобучение.
        n_jobs=-1,                # Количество потоков (CPU). -1 = использовать все доступные.
        random_state=42,           # Фиксируем зерно генератора случайных чисел для воспроизводимости результата.
        min_child_weight=1.0006
    )
    return model


def model_testing(model, data):
    print("Начинаем обучение модели...")

    if data is None or data.empty:
        print("Нет данных для обучения!")
        return None
    data = data.dropna(subset=["hour"])

    X = data.drop(columns=["hour"])
    y = data["hour"].astype(int)  # Целевая переменная — час пика

    # Удаляем редкие классы (< 2 наблюдений)
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    # Разделение с сохранением пропорций классов
    train_size = int(len(X) * 0.8)
    X_train, X_valid = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_valid = y.iloc[:train_size], y.iloc[train_size:]
    sample_weight = np.where(X_train["is_imputed"] == 1, 0.7, 1)
    recency_weight = np.exp(0.00001 * np.arange(len(X_train)))  # фокус на более свежие данные
    sample_weight = sample_weight * recency_weight
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        sample_weight=sample_weight,
        callbacks=[
            early_stopping(stopping_rounds=3000), 
            log_evaluation(period=200) 
        ]
    )

    y_pred = model.predict(X_valid)
    plot_confusion_matrix_and_metrics(model, X_valid, y_valid)
    acc = accuracy_score(y_valid, y_pred)
    return model, acc

#функция для проверки точности уже обученной модели
def model_accuracy(model, data):
    print("Начинаем тестирование модели...")

    if model is None or data is None or data.empty:
        print("Нет модели или данных для тестирования!")
        return None
        
    data = data.dropna(subset=["hour"])
    X = data.drop(columns=["hour"])
    y = data["hour"].astype(int)

    y_pred = model.predict(X)
    plot_confusion_matrix_and_metrics(model, X, y)
    acc = accuracy_score(y, y_pred)
    return acc

def predict_month(model, month, year=2025):
    # Загрузка исторических данных
    historical_data_2021_2024 = load_data("data/data_2022-2024.csv", start_date="2022-01-01")
    historical_data_2025 = load_data("data/data_2025.csv", start_date="2025-01-01")
    historical_data_2021_2024 = pd.concat([historical_data_2021_2024, historical_data_2025[historical_data_2025['date'] < pd.Timestamp(f"{year}-{month:02d}-01")]], ignore_index=True)
    last_df = historical_data_2021_2024.copy()  # Копируем для обновлений

    # Даты от января до конца переданного месяца
    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
    all_dates = pd.date_range(start_date, end_date)

    predictions = []
    pred_dates = []

    for day in all_dates:
        # Добавляем новый день с NaN hour
        new_row = pd.DataFrame({'date': [day], 'hour': [np.nan]})
        last_df = pd.concat([last_df, new_row], ignore_index=True)

        # Применяем process_data
        processed = process_data(last_df.copy())  # copy, чтобы не мутировать

        # Берем последнюю строку для предсказания
        X_new = processed.iloc[-1:].drop(columns=['hour'])

        # Предсказание
        pred = model.predict(X_new)[0]
        predictions.append(pred)
        pred_dates.append(day)

        # Обновляем last_df с предсказанием
        last_df.at[last_df.index[-1], 'hour'] = pred

    # Результат: только для переданного месяца
    result = pd.DataFrame({'date': pred_dates, 'predicted_hour': predictions})
    result = result[result['date'].dt.month == month]

    # Сохранение в CSV
    csv_file = f"data/predictions_{year}_{month:02d}.csv"
    #result.to_csv(csv_file, index=False)
    print(f"Предсказания сохранены в {csv_file}")
    return result