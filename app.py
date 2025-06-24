import streamlit as st
import numpy as np

from ai_core.model_handler import ModelHandler
from ai_core.data_utils import detect_column_type, validate_numeric
from ai_core.history_logger import save_history
from web_ui.ui_components import plot_regression_line

st.set_page_config(page_title="🧠 Мини-ИИ", layout="centered")

st.title("🧠 Мини-ИИ: Линейная регрессия")
st.write("Введите данные и обучите модель на лету!")

model_handler = ModelHandler()

# Ввод данных
X = []
y = []

num_vars = st.number_input("Количество входных переменных (например, x, z):", min_value=1, value=1)

if 'points' not in st.session_state:
    st.session_state.points = []

col_titles = ["x", "z", "a", "b", "c", "d", "e", "f"]

for i in range(num_vars + 1):
    col_titles[i % len(col_titles)]

cols = st.columns(num_vars + 1)
inputs = []

for i in range(num_vars):
    with cols[i]:
        val = st.number_input(f"{col_titles[i]}", key=f"input_{i}")
        inputs.append(val)

with cols[-1]:
    y_val = st.number_input("y", key="y_input")
    inputs.append(y_val)

if st.button("Добавить точку"):
    if all(validate_numeric(v) for v in inputs):
        x_vals = inputs[:-1]
        y_val = inputs[-1]
        X.append(x_vals)
        y.append(y_val)
        st.session_state.points.append((x_vals, y_val))
        st.success("✅ Точка добавлена!")
    else:
        st.error("Проверьте значения — все должны быть числами.")

if len(X) >= 2 and st.button("Обучить модель"):
    X_np = np.array(X)
    y_np = np.array(y)
    coef, intercept = model_handler.train(X_np, y_np)
    st.success(f"Модель обучена: y = {coef} * x + {intercept}")
    save_history(X_np, y_np, coef, intercept)

    plot_regression_line(X_np[:, 0], y_np, model_handler.model)

# Предсказание
if model_handler.model is not None:
    st.subheader("🔮 Предсказание")
    pred_inputs = []
    cols_pred = st.columns(num_vars)
    for i in range(num_vars):
        with cols_pred[i]:
            val = st.number_input(f"x{i+1}", key=f"pred_input_{i}")
            pred_inputs.append(val)
    if st.button("Сделать предсказание"):
        prediction = model_handler.predict([pred_inputs])
        st.info(f"Предсказанное значение: {prediction[0]:.2f}")