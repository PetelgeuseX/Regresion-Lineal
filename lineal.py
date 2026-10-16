import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ─────────────────────────────────────────────────────────────
# Configuración general y estilo profesional
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Regresión Lineal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1f2a44;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px 16px;
        }
        .interpretation-box {
            background-color: #f0f4ff;
            border-left: 4px solid #3b5bdb;
            border-radius: 6px;
            padding: 14px 18px;
            margin-top: 10px;
        }
        .reliability-high { color: #2b8a3e; font-weight: 700; }
        .reliability-medium { color: #e67700; font-weight: 700; }
        .reliability-low { color: #c92a2a; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">📈 Simulador de Regresión Lineal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Carga un dataset (o usa uno de ejemplo), entrena el modelo, '
    'visualiza los resultados y realiza predicciones en tiempo real.</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_dataset_ejemplo(nombre: str) -> pd.DataFrame:
    """Carga datasets reales. Intenta descargar y, si falla, usa los incluidos en scikit-learn."""
    if nombre == "Combustible (Auto MPG - UCI)":
        try:
            return pd.read_csv(
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
            )
        except Exception:
            st.warning("No se pudo descargar el dataset remoto. Usando Diabetes como alternativa.")
            nombre = "Diabetes (clínico)"

    if nombre == "Propinas (tips)":
        try:
            return pd.read_csv(
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
            )
        except Exception:
            st.warning("No se pudo descargar el dataset remoto. Usando Diabetes como alternativa.")
            nombre = "Diabetes (clínico)"

    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": "progresion_diabetes"})
    return df


def interpretar_r2(r2: float):
    """Devuelve etiqueta, clase CSS y texto de confiabilidad según el R²."""
    if r2 >= 0.85:
        return (
            "Alta",
            "reliability-high",
            "El modelo explica la mayor parte de la variabilidad de los datos. "
            "Las predicciones son **confiables** dentro del rango observado.",
        )
    if r2 >= 0.60:
        return (
            "Moderada",
            "reliability-medium",
            "El modelo captura una parte importante de la tendencia, pero existe "
            "variación no explicada. Las predicciones son **útiles como estimación**, "
            "no como valor exacto.",
        )
    if r2 > 0.0:
        return (
            "Baja",
            "reliability-low",
            "El modelo explica poca variabilidad de la variable objetivo. "
            "Las predicciones tienen **baja confiabilidad**; considera agregar "
            "más variables o probar otro tipo de modelo.",
        )
    return (
        "Nula",
        "reliability-low",
        "El R² es negativo: el modelo predice **peor que la media** de los datos. "
        "La relación entre las variables elegidas no es lineal o no aporta información útil.",
    )


def interpretar_metricas(mae, rmse, y_real):
    rango = np.max(y_real) - np.min(y_real)
    if rango == 0:
        return "—"
    porcentaje = (rmse / rango) * 100
    return (
        f"El error promedio de las predicciones (MAE) es **{mae:.3f}** unidades de la variable objetivo. "
        f"El RMSE ({rmse:.3f}) equivale al **{porcentaje:.1f}%** del rango total de los datos, "
        f"lo que indica el tamaño típico de un error de predicción."
    )


def grafica_interactiva_2d(modelo, X, y, X_test, y_pred, x_name, y_name, r2, fijas=None):
    """Gráfica 2D interactiva: datos reales, recta de regresión y predicciones.

    fijas: variables extra (regresión múltiple) fijadas en su valor medio para trazar la recta.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[x_name], y=y, mode="markers", name="Datos reales",
        marker=dict(color="#74a9cf", size=8, opacity=0.6),
        hovertemplate=f"{x_name}=%{{x:.3f}}<br>{y_name}=%{{y:.3f}}<extra></extra>",
    ))
    eje_x = np.linspace(X[x_name].min(), X[x_name].max(), 200)
    linea = pd.DataFrame({x_name: eje_x})
    if fijas:
        for c, v in fijas.items():
            linea[c] = v
    fig.add_trace(go.Scatter(
        x=eje_x, y=modelo.predict(linea), mode="lines", name=f"Regresión (R²={r2:.3f})",
        line=dict(color="#d62728", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=X_test[x_name], y=y_pred, mode="markers", name="Predicciones (test)",
        marker=dict(color="#2ca02c", size=11, symbol="x",
                    line=dict(color="black", width=1)),
        hovertemplate=f"{x_name}=%{{x:.3f}}<br>predicho=%{{y:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        title="Regresión lineal (2D interactiva)",
        xaxis_title=x_name, yaxis_title=y_name,
        template="plotly_white", height=520,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=60, b=60),
    )
    return fig


def grafica_interactiva_3d(modelo, X, y, X_test, y_pred, x1, x2, y_name, r2, fijas=None):
    """Gráfica 3D interactiva: nube de puntos + plano de regresión (arrastra para rotar).

    fijas: variables adicionales fijadas en su valor medio (cuando hay más de 2 predictores).
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[x1], y=X[x2], z=y, mode="markers", name="Datos reales",
        marker=dict(color=y, colorscale="Blues", size=5, opacity=0.75,
                    colorbar=dict(title=y_name)),
        hovertemplate=f"{x1}=%{{x:.3f}}<br>{x2}=%{{y:.3f}}<br>{y_name}=%{{z:.3f}}<extra></extra>",
    ))

    x1_range = np.linspace(X[x1].min(), X[x1].max(), 25)
    x2_range = np.linspace(X[x2].min(), X[x2].max(), 25)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)

    plane_input = pd.DataFrame({x1: xx1.ravel(), x2: xx2.ravel()})
    if fijas:
        for col, val in fijas.items():
            plane_input[col] = val
    zz = modelo.predict(plane_input).reshape(xx1.shape)

    fig.add_trace(go.Surface(
        x=x1_range, y=x2_range, z=zz, opacity=0.45,
        colorscale="Reds", showscale=False, name="Plano de regresión",
        hovertemplate=f"{x1}=%{{x:.3f}}<br>{x2}=%{{y:.3f}}<br>ajuste=%{{z:.3f}}<extra></extra>",
    ))

    fig.add_trace(go.Scatter3d(
        x=X_test[x1], y=X_test[x2], z=y_pred, mode="markers", name="Predicciones (test)",
        marker=dict(color="#2ca02c", size=6, symbol="x",
                    line=dict(color="black", width=1)),
        hovertemplate=f"{x1}=%{{x:.3f}}<br>{x2}=%{{y:.3f}}<br>predicho=%{{z:.3f}}<extra></extra>",
    ))

    titulo = f"Plano de regresión 3D (R²={r2:.3f}) — arrastra para rotar"
    if fijas:
        vars_fijas = ", ".join(f"{k}={v:.2f}" for k, v in fijas.items())
        titulo += f" · fijas: {vars_fijas}"

    fig.update_layout(
        title=titulo,
        scene=dict(xaxis_title=x1, yaxis_title=x2, zaxis_title=y_name),
        template="plotly_white", height=580,
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=70, b=20),
    )
    return fig


def grafica_dispersion_interactiva(x, y, xlab, ylab, titulo, color="#5e60ce", ref_line=False):
    """Scatter interactivo para reales vs predichos y residuales."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=color, size=9, opacity=0.65),
        hovertemplate=f"{xlab}=%{{x:.3f}}<br>{ylab}=%{{y:.3f}}<extra></extra>",
    ))
    if ref_line:
        lims = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines", name="Predicción perfecta",
            line=dict(color="#c92a2a", width=2, dash="dash"),
        ))
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="#c92a2a", line_width=2)
    fig.update_layout(
        title=titulo, xaxis_title=xlab, yaxis_title=ylab,
        template="plotly_white", height=420, showlegend=ref_line,
        margin=dict(t=50, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Barra lateral: datos
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Datos")
    fuente = st.radio(
        "Origen del dataset",
        ["Dataset de ejemplo (real)", "Subir archivo CSV"],
    )

    df = None
    if fuente == "Dataset de ejemplo (real)":
        opcion = st.selectbox(
            "Dataset",
            [
                "Combustible (Auto MPG - UCI)",
                "Diabetes (clínico)",
                "Propinas (tips)",
            ],
            help="Datasets reales populares para regresión.",
        )
        df = cargar_dataset_ejemplo(opcion)
    else:
        archivo = st.file_uploader("Archivo CSV", type=["csv"])
        if archivo is not None:
            try:
                df = pd.read_csv(archivo)
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

    if df is not None:
        st.success(f"Dataset cargado: **{df.shape[0]}** filas × **{df.shape[1]}** columnas")

if df is None:
    st.info("👈 Selecciona o sube un dataset desde la barra lateral para comenzar.")
    st.stop()

# Limpieza básica
df = df.dropna(how="all").reset_index(drop=True)
columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

if len(columnas_numericas) < 2:
    st.error("El dataset necesita al menos 2 columnas numéricas para entrenar un modelo de regresión lineal.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Vista previa y estadísticas
# ─────────────────────────────────────────────────────────────
tab_preview, tab_config, tab_modelo = st.tabs(
    ["👁 Vista previa de datos", "⚙ Configuración del modelo", "🎯 Resultados y predicción"]
)

with tab_preview:
    st.subheader("Muestra de los datos")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(df[columnas_numericas].describe().T.round(3), use_container_width=True)

    with st.expander("Ver información de columnas"):
        info = pd.DataFrame({
            "Tipo": df.dtypes.astype(str),
            "No nulos": df.notna().sum(),
            "Nulos": df.isna().sum(),
            "Únicos": df.nunique(),
        })
        st.dataframe(info, use_container_width=True)

with tab_config:
    st.subheader("Selección de variables")

    col_x, col_y = st.columns([3, 2])
    with col_x:
        variables_x = st.multiselect(
            "Variables de entrada (predictoras)",
            columnas_numericas,
            default=[c for c in columnas_numericas[:1]],
            help="Puedes seleccionar una (regresión simple) o varias (regresión múltiple).",
        )
    with col_y:
        variables_y = st.selectbox(
            "Variable a predecir (objetivo)",
            [c for c in columnas_numericas if c not in variables_x] or columnas_numericas,
        )

    if variables_y in variables_x:
        st.warning("La variable objetivo no puede estar también entre las variables de entrada.")
        variables_x = [c for c in variables_x if c != variables_y]

    st.divider()
    st.subheader("Parámetros de entrenamiento")

    col_a, col_b = st.columns(2)
    with col_a:
        test_size = st.slider(
            "Proporción de datos de prueba",
            0.10, 0.40, 0.20, 0.05,
            help="Porcentaje de datos reservados para evaluar el modelo ( típicamente 20% ).",
        )
    with col_b:
        semilla = st.number_input("Semilla aleatoria (reproducibilidad)", 0, 9999, 42)

    entrenar = st.button("🚀 Entrenar modelo", type="primary", use_container_width=True)

    if entrenar and variables_x:
        df_modelo = df[variables_x + [variables_y]].dropna()
        X = df_modelo[variables_x]
        y = df_modelo[variables_y]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(semilla)
        )

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)

        st.session_state.update({
            "modelo": modelo,
            "variables_x": variables_x,
            "variables_y": variables_y,
            "mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
            "X": X, "y": y,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "y_pred": y_pred,
            "df_modelo": df_modelo,
        })

with tab_modelo:
    if "modelo" not in st.session_state:
        st.info("👈 Ve a la pestaña **Configuración del modelo**, elige las variables y pulsa **Entrenar modelo**.")
        st.stop()

    m = st.session_state
    modelo, variables_x, variables_y = m["modelo"], m["variables_x"], m["variables_y"]
    y_test, y_pred = m["y_test"], m["y_pred"]
    rango_total = np.max(m["y"]) - np.min(m["y"])

    # ── Métricas ────────────────────────────────────────────
    st.subheader("📊 Métricas de evaluación (conjunto de prueba)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{m['mae']:.4f}", help="Error absoluto promedio")
    c2.metric("MSE", f"{m['mse']:.4f}", help="Error cuadrático medio")
    c3.metric("RMSE", f"{m['rmse']:.4f}", help="Raíz del error cuadrático medio")
    c4.metric("R²", f"{m['r2']:.4f}", help="Coeficiente de determinación")

    nivel, clase, texto_conf = interpretar_r2(m["r2"])
    st.markdown(
        f"""
        <div class="interpretation-box">
            <b>Interpretación de resultados</b><br><br>
            <b>Confiabilidad de las predicciones:</b>
            <span class="{clase}">{nivel}</span><br>
            {texto_conf}<br><br>
            {interpretar_metricas(m["mae"], m["rmse"], y_test.values)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Coeficientes ────────────────────────────────────────
    with st.expander("Ver coeficientes del modelo (ecuación de la regresión)"):
        coefs = pd.DataFrame({
            "Variable": variables_x,
            "Coeficiente": modelo.coef_,
        }).sort_values("Coeficiente", key=abs, ascending=False)
        st.dataframe(coefs.style.format({"Coeficiente": "{:.4f}"}), use_container_width=True)
        ecuacion = f"**{variables_y}** = {modelo.intercept_:.4f}"
        for var, coef in zip(variables_x, modelo.coef_):
            signo = "+" if coef >= 0 else "-"
            ecuacion += f" {signo} {abs(coef):.4f}·{var}"
        st.markdown(ecuacion)

    st.divider()

    # ── Gráficas ────────────────────────────────────────────
    st.subheader("📉 Visualización de resultados")

    opciones_modo = ["Gráfica estática 2D", "Interactiva 2D (zoom y hover)"]
    if len(variables_x) >= 2:
        opciones_modo.append("Interactiva 3D (plano rotable)")
    modo = st.radio("Modo de visualización", opciones_modo, horizontal=True)

    residuales = y_test.values - y_pred

    if modo == "Gráfica estática 2D":
        plt.style.use("seaborn-v0_8-whitegrid")

        # 1) Regresión (1 variable) / reales vs predichos (múltiple)
        fig1, ax1 = plt.subplots(figsize=(7, 4.5))
        if len(variables_x) == 1:
            x_name = variables_x[0]
            ax1.scatter(m["X"], m["y"], color="#74a9cf", alpha=0.55, s=45, label="Datos reales")
            eje_x = np.linspace(m["X"][x_name].min(), m["X"][x_name].max(), 300).reshape(-1, 1)
            linea = pd.DataFrame(eje_x, columns=[x_name])
            ax1.plot(eje_x, modelo.predict(linea), color="#d62728", linewidth=2.4,
                     label=f"Regresión (R²={m['r2']:.3f})")
            ax1.scatter(m["X_test"], y_pred, color="#2ca02c", marker="X", s=70,
                        edgecolors="black", linewidths=0.5, label="Valores predichos (test)")
            ax1.set_xlabel(x_name)
            ax1.set_ylabel(variables_y)
            ax1.set_title("Regresión lineal simple: datos reales, recta ajustada y predicciones", fontsize=12)
        else:
            ax1.scatter(m["X"].iloc[:, 0], m["y"], color="#74a9cf", alpha=0.55, s=45,
                        label=f"Datos reales (1.ª variable: {variables_x[0]})")
            ax1.scatter(m["X_test"].iloc[:, 0], y_pred, color="#d62728", marker="X", s=70,
                        edgecolors="black", linewidths=0.5, label="Valores predichos (test)")
            ax1.set_xlabel(variables_x[0])
            ax1.set_ylabel(variables_y)
            ax1.set_title("Regresión múltiple: reales vs predichos (por 1.ª variable)", fontsize=12)
        ax1.legend()
        st.pyplot(fig1)

        # 2) Reales vs predichos + 3) Residuales
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig2, ax2 = plt.subplots(figsize=(5.5, 4))
            ax2.scatter(y_test, y_pred, color="#5e60ce", alpha=0.65, s=50)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax2.plot(lims, lims, "--", color="#c92a2a", linewidth=1.8, label="Predicción perfecta")
            ax2.set_xlabel("Valor real")
            ax2.set_ylabel("Valor predicho")
            ax2.set_title("Reales vs Predichos")
            ax2.legend()
            st.pyplot(fig2)
        with col_g2:
            fig3, ax3 = plt.subplots(figsize=(5.5, 4))
            ax3.scatter(y_pred, residuales, color="#f08c00", alpha=0.65, s=50)
            ax3.axhline(0, color="#c92a2a", linestyle="--", linewidth=1.8)
            ax3.set_xlabel("Valor predicho")
            ax3.set_ylabel("Residual (real − predicho)")
            ax3.set_title("Gráfica de residuales")
            st.pyplot(fig3)
    else:
        # Modos interactivos (Plotly)
        if modo == "Interactiva 3D (plano rotable)":
            importancias = np.abs(modelo.coef_)
            idx_top2 = np.argsort(importancias)[-2:]
            x1, x2 = variables_x[idx_top2[0]], variables_x[idx_top2[1]]
            fijas = None
            if len(variables_x) > 2:
                otras = [c for c in variables_x if c not in (x1, x2)]
                fijas = {c: float(m["X"][c].mean()) for c in otras}
            st.plotly_chart(
                grafica_interactiva_3d(modelo, m["X"], m["y"], m["X_test"], y_pred,
                                       x1, x2, variables_y, m["r2"], fijas=fijas),
                use_container_width=True,
            )
            if fijas:
                st.info(
                    f"Con más de 2 predictores el gráfico 3D usa las 2 variables más influyentes "
                    f"(**{x1}**, **{x2}**) y mantiene las demás en su valor promedio."
                )
        else:
            fijas = None
            if len(variables_x) > 1:
                fijas = {c: float(m["X"][c].mean()) for c in variables_x[1:]}
            st.plotly_chart(
                grafica_interactiva_2d(modelo, m["X"], m["y"], m["X_test"], y_pred,
                                       variables_x[0], variables_y, m["r2"], fijas=fijas),
                use_container_width=True,
            )
            if len(variables_x) > 1:
                st.caption(
                    f"Regresión múltiple: la recta se traza variando **{variables_x[0]}** y "
                    "manteniendo las demás variables en su promedio."
                )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(
                grafica_dispersion_interactiva(y_test, y_pred, "Valor real", "Valor predicho",
                                               "Reales vs Predichos", color="#5e60ce", ref_line=True),
                use_container_width=True,
            )
        with col_g2:
            st.plotly_chart(
                grafica_dispersion_interactiva(y_pred, residuales, "Valor predicho",
                                               "Residual (real − predicho)", "Gráfica de residuales",
                                               color="#f08c00"),
                use_container_width=True,
            )

    st.caption(
        "Los residuales deberían distribuirse de forma aleatoria alrededor de 0. "
        "Si muestran un patrón (curva, embudo), la relación no es puramente lineal. "
        "En los modos interactivos puedes hacer zoom, ver valores al pasar el mouse "
        "y, en 3D, arrastrar para rotar el plano."
    )

    st.divider()

    # ── Predicción con nuevos valores ───────────────────────
    st.subheader("🔮 Realizar una nueva predicción")
    st.markdown("Introduce valores para las variables de entrada y obtén la estimación del modelo.")

    entradas = {}
    cols = st.columns(min(len(variables_x), 4))
    for i, var in enumerate(variables_x):
        vmin, vmax = float(m["X"][var].min()), float(m["X"][var].max())
        promedio = float(m["X"][var].mean())
        with cols[i % len(cols)]:
            entradas[var] = st.number_input(
                var, value=round(promedio, 3), format="%.4f",
                help=f"Rango observado: {vmin:.2f} a {vmax:.2f}",
            )

    if st.button("Predecir", type="primary"):
        entrada_df = pd.DataFrame([entradas])[variables_x]
        resultado = modelo.predict(entrada_df)[0]
        st.success(f"Predicción de **{variables_y}**: **{resultado:.4f}**")
        desv = float(np.std(m["y"] - modelo.predict(m["X"])))
        st.caption(
            f"Con un error típico aproximado de ±{desv:.3f} unidades "
            f"(desviación estándar de los residuales del modelo)."
        )
