import pandas as pd
import numpy as np
import holidays
DATA_PATHS = ["data/data_2023.csv", "data/data_2024.csv", "data/data_2025.csv", "data/data_2023-2024.csv", "data/data_2022-2024.csv"]

import matplotlib.pyplot as plt

def target_distribution_by_year(data):
    data["year"] = pd.to_datetime(data["date"]).dt.year
    print(data["hour"].unique())
    dist = data.groupby("year")["hour"].value_counts(normalize=True).unstack().fillna(0)
    
    dist.T.plot(kind="bar", figsize=(15,6), width=0.8)
    plt.title("Распределение целевой переменной (часа) по годам")
    plt.ylabel("Доля")
    plt.show()

def target_distribution_by_month(data):
    data["month"] = pd.to_datetime(data["date"]).dt.month
    unique_months = sorted(data["month"].unique())
    n_months = len(unique_months)
    
    # Сетка подграфиков: 4 ряда по 3 столбца для 12 месяцев
    fig, axes = plt.subplots(nrows=(n_months + 2) // 3, ncols=3, figsize=(15, 5 * ((n_months + 2) // 3)), squeeze=False)
    axes = axes.flatten()
    
    for i, month in enumerate(unique_months):
        month_data = data[data["month"] == month]
        dist = month_data["hour"].value_counts(normalize=True).sort_index().reindex(range(24), fill_value=0)
        
        axes[i].bar(dist.index, dist.values, width=0.8, edgecolor='black')
        axes[i].set_title(f"Month {month}")
        axes[i].set_xlabel("Hour")
        axes[i].set_ylabel("Proportion")
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].grid(True, alpha=0.3)
    
    # Удаляем лишние оси
    for j in range(n_months, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def target_distribution_by_dayofweek(data):
    data["dow"] = pd.to_datetime(data["date"]).dt.dayofweek
    unique_dows = sorted(data["dow"].unique())
    n_dows = len(unique_dows)
    
    # Сетка подграфиков: 3 ряда по 3 столбца для 7 дней недели
    fig, axes = plt.subplots(nrows=(n_dows + 2) // 3, ncols=3, figsize=(15, 5 * ((n_dows + 2) // 3)), squeeze=False)
    axes = axes.flatten()
    
    for i, dow in enumerate(unique_dows):
        dow_data = data[data["dow"] == dow]
        dist = dow_data["hour"].value_counts(normalize=True).sort_index().reindex(range(24), fill_value=0)
        
        axes[i].bar(dist.index, dist.values, width=0.8, edgecolor='black')
        axes[i].set_title(f"Day of Week {dow} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]})")
        axes[i].set_xlabel("Hour")
        axes[i].set_ylabel("Proportion")
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].grid(True, alpha=0.3)
    
    # Удаляем лишние оси
    for j in range(n_dows, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def load_data(file_path, start_date="2021-01-01") -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path, sep='\s+', header=None)
        data.columns = ["date", "hour"]
        
        # Преобразуем дату в datetime
        data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")

        # Создаём полный календарь от стартовой до последней даты в данных
        start_date = pd.Timestamp(start_date)
        end_date = data["date"].max()
        full_dates = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})

        # Объединяем с исходными данными (пропущенные будут NaN)
        data = pd.merge(full_dates, data, on="date", how="left")
        data["is_imputed"] = data["hour"].isna().astype(int)

        
        # Добавляем временные фичи для заполнения
        data["dow"] = data["date"].dt.weekday
        data["month"] = data["date"].dt.month
        # Заполняем пропуски медианой по dow и month
        data["hour"] = data.groupby(['dow', 'month'])['hour'].transform(lambda x: x.fillna(np.round(x.median())))
        # Если остались NaN (например, для редких групп), используем global median
        data["hour"] = data["hour"].fillna(np.round(data["hour"].median()))
        # Удаляем временные колонки
        data.drop(columns=["dow", "month"], inplace=True)
        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    if data is None:
        return None
    
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
    data = data.sort_values("date").reset_index(drop=True)
    data["dow"] = data["date"].dt.weekday                    # день недели (0=Пн,...,6=Вс)
    data["month"] = data["date"].dt.month                    # месяц
    data["day"] = data["date"].dt.day                        # день месяца
    data["year"] = data["date"].dt.year                      # год
    data["weekofyear"] = data["date"].dt.isocalendar().week.astype(int)  # номер недели
    data["is_weekend"] = (data["dow"] >= 5).astype(int)       # выходной (сб/вс)
    data["is_month_start"] = data["date"].dt.is_month_start.astype(int)  # начало месяца
    data["is_month_end"] = data["date"].dt.is_month_end.astype(int)      # конец месяца
    data["is_quarter_start"] = data["date"].dt.is_quarter_start.astype(int)  # начало квартала
    data["is_quarter_end"] = data["date"].dt.is_quarter_end.astype(int)      # конец квартала  
    data["is_year_start"] = data["date"].dt.is_year_start.astype(int)        # начало года
    data["is_year_end"] = data["date"].dt.is_year_end.astype(int)            # конец года

    # Лаги по метке
    data["hour_yesterday"] = data["hour"].shift(1)
    data["hour_2days_ago"] = data["hour"].shift(2)
    data["hour_7days_ago"] = data["hour"].shift(7)
    data["hour_14days_ago"] = data["hour"].shift(14)

    
    data["hour_unique_7d"] = (
        data["hour"].shift(1).rolling(window=7).apply(lambda x: len(np.unique(x)))
    )
    
    ru = holidays.CountryHoliday("RU")
    data["is_holiday"] = data["date"].isin(ru).astype(int)

    # разница в моде между предыдущим годом и годом перед ним по месяцам
    data['prev_year'] = data['year'] - 1
    data['two_years_ago'] = data['year'] - 2
    
    # Вычисляем моду по году и месяцу
    monthly_modes = data.groupby(['year', 'month'])['hour'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    ).reset_index(name='month_mode')
    
    # Добавляем моду предыдущего года
    data = data.merge(
        monthly_modes.rename(columns={'year': 'prev_year', 'month_mode': 'prev_year_month_mode'}),
        on=['month', 'prev_year'],
        how='left'
    )
    
    # Добавляем моду два года назад
    data = data.merge(
        monthly_modes.rename(columns={'year': 'two_years_ago', 'month_mode': 'two_years_ago_month_mode'}),
        on=['month', 'two_years_ago'],
        how='left'
    )
    
    # Вычисляем разницу в моде (prev_year - two_years_ago)
    data['historical_mode_diff'] = data['prev_year_month_mode'] - data['two_years_ago_month_mode']
    
    # Заполняем NaN для случаев, когда данных за прошлые годы нет
    data['prev_year_month_mode'] = data['prev_year_month_mode'].fillna(data['hour'].median())
    data['two_years_ago_month_mode'] = data['two_years_ago_month_mode'].fillna(data['hour'].median())
    data['historical_mode_diff'] = data['historical_mode_diff'].fillna(0)
    
    # Удаляем вспомогательные колонки
    data.drop(columns=['year', 'prev_year', 'two_years_ago', 'prev_year_month_mode', 'two_years_ago_month_mode'], inplace=True)

    # Rolling фичи (shift(1) чтобы избежать leakage)
    data['hour_mean_7d'] = data['hour'].shift(1).rolling(7).mean()
    data['hour_std_7d'] = data['hour'].shift(1).rolling(7).std()
    data['hour_median_30d'] = data['hour'].shift(1).rolling(30).median()
    data['hour_trend_365d'] = data['hour'] - data['hour'].shift(365)  # Годовой тренд (diff)

    # Заполнение NaN
    lag_cols = ['hour_mean_7d', 'hour_std_7d', 'hour_median_30d', 'hour_trend_365d']
    for col in lag_cols:
        data[col] = data[col].fillna(np.round(data['hour'].median()))
    data.drop(columns=["date"], inplace=True)
    return data
