import streamlit as st
import matplotlib.pyplot as plt

def plot_regression_line(X, y, model):
    if X.shape[1] == 1:
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Данные')
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        ax.plot(x_range, y_pred, color='red', label='Модель')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("График пока строится только для одной переменной.")