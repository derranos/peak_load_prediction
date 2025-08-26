import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import joblib  # Для загрузки модели
from model_funcs.model_funcs import predict_month

class PredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Peak Hour Prediction")
        self.geometry("600x400")

        # Заголовок
        tk.Label(self, text="Predict Peak Hours for 2025", font=("Arial", 16)).pack(pady=10)

        # Выбор месяца
        self.month_var = tk.StringVar()
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month_combo = ttk.Combobox(self, values=months, textvariable=self.month_var)
        month_combo.pack(pady=10)

        # Кнопки
        tk.Button(self, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=10)
        tk.Button(self, text="Save to CSV", command=self.save).pack(side=tk.LEFT, padx=10)

        # Таблица (ScrolledText для простоты)
        self.result_text = ScrolledText(self, height=10)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=10)

        # Статус
        self.status = tk.Label(self, text="Ready")
        self.status.pack(pady=5)

        # Модель (замени на реальную)
        self.model = joblib.load("model_funcs/lightgbm_model.pkl")  # Или lgb.Booster(model_file="model_funcs/lightgbm_model.txt")
        self.predictions = None

    def predict(self):
        month_name = self.month_var.get()
        if not month_name:
            messagebox.showerror("Error", "Select month!")
            return
        month_num = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"].index(month_name) + 1
        self.status.config(text="Predicting...")
        try:
            self.predictions = predict_month(self.model, month_num, year=2025)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, self.predictions.to_string(index=False))
            self.status.config(text="Done")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Error")

    def save(self):
        if self.predictions is None:
            messagebox.showerror("Error", "No predictions!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv")
        if file_path:
            self.predictions.to_csv(file_path, index=False)
            self.status.config(text="Saved")

if __name__ == "__main__":
    app = PredictionApp()
    app.mainloop()