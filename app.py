import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# --- 1. Título y Descripción de la Aplicación ---
st.set_page_config(layout="wide", page_title="Predicción de Indicadores Financieros")
st.title("🔮 Predicción de Indicadores Financieros con LSTM")
st.markdown("""
Sube un archivo Excel con los datos históricos de los indicadores financieros de las cooperativas.
La aplicación te permitirá seleccionar un RUC y predecir el comportamiento futuro de sus indicadores.
""")

# --- 2. Carga de Modelo y Escalador (con cache y manejo de errores) ---
@st.cache_resource
def load_prediction_assets():
    with st.spinner("Cargando modelo de predicción y escalador..."):
        try:
            model_path = "modelo_general_lstm.h5"
            scaler_path = "scaler_general.pkl"

            if not os.path.exists(model_path):
                st.error(f"❌ Error: Archivo del modelo '{model_path}' no encontrado. Asegúrate de que esté en el mismo directorio de 'app.py'.")
                return None, None
            if not os.path.exists(scaler_path):
                st.error(f"❌ Error: Archivo del escalador '{scaler_path}' no encontrado. Asegúrate de que esté en el mismo directorio de 'app.py'.")
                return None, None

            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            st.success("✅ Modelo y escalador cargados correctamente.")
            return model, scaler
        except Exception as e:
            st.error(f"❌ Error al cargar modelo o escalador: {e}. Por favor, verifica los archivos.")
            return None, None

model, scaler = load_prediction_assets()

# Si el modelo o escalador no cargaron, detenemos la app
if model is None or scaler is None:
    st.stop()

# Definición de indicadores esperados por el modelo
indicadores_elegidos_raw = [
    'MOROSIDAD DE LA CARTERA TOTAL',
    'COBERTURA DE LA CARTERA PROBLEMÁTICA',
    'RESULTADOS DEL EJERCICIO / ACTIVO PROMEDIO',
    'FONDOS DISPONIBLES / TOTAL DEPOSITOS A CORTO PLAZO ',
    'RESULTADOS DEL EJERCICIO / PATRIMONIO PROMEDIO'
]
indicadores_renamed = {
    'MOROSIDAD DE LA CARTERA TOTAL': 'MOROSIDAD',
    'COBERTURA DE LA CARTERA PROBLEMÁTICA': 'COBERTURA_CARTERA_PROBLEMATICA',
    'RESULTADOS DEL EJERCICIO / ACTIVO PROMEDIO': 'ROA',
    'FONDOS DISPONIBLES / TOTAL DEPOSITOS A CORTO PLAZO ': 'LIQUIDEZ',
    'RESULTADOS DEL EJERCICIO / PATRIMONIO PROMEDIO': 'ROE'
}
# Orden final de las columnas después del pivoteo, tal como el scaler lo espera
indicadores_model_order = ['COBERTURA_CARTERA_PROBLEMATICA', 'LIQUIDEZ', 'MOROSIDAD', 'ROA', 'ROE']


# --- 3. Instrucciones y Ejemplo de Formato de Archivo ---
st.header("Formato del Archivo Excel Requerido:")
st.markdown("""
El archivo Excel (`.xlsx`) debe contener los siguientes 4 columnas con datos históricos:

-   `RUC`: Identificador de la cooperativa (ej. '1090000001001').
-   `FECHA_CORTE`: Fecha de los datos (ej. '2023-01-31').
-   `INDICADOR_FINANCIERO`: Nombre del indicador (debe coincidir con los esperados).
-   `VALOR`: Valor numérico del indicador.

**Nombres de Indicadores esperados en la columna 'INDICADOR_FINANCIERO':**
-   `MOROSIDAD DE LA CARTERA TOTAL`
-   `COBERTURA DE LA CARTERA PROBLEMÁTICA`
-   `RESULTADOS DEL EJERCICIO / ACTIVO PROMEDIO`
-   `FONDOS DISPONIBLES / TOTAL DEPOSITOS A CORTO PLAZO `
-   `RESULTADOS DEL EJERCICIO / PATRIMONIO PROMEDIO`

**Ejemplo de las primeras filas de tu archivo Excel:**
""")

example_data = {
    'RUC': ['1090000001001', '1090000001001', '1090000001001', '1090000001001', '1090000001001'],
    'FECHA_CORTE': ['2023-01-31', '2023-01-31', '2023-01-31', '2023-01-31', '2023-01-31'],
    'INDICADOR_FINANCIERO': [
        'MOROSIDAD DE LA CARTERA TOTAL',
        'COBERTURA DE LA CARTERA PROBLEMÁTICA',
        'RESULTADOS DEL EJERCICIO / ACTIVO PROMEDIO',
        'FONDOS DISPONIBLES / TOTAL DEPOSITOS A CORTO PLAZO ',
        'RESULTADOS DEL EJERCICIO / PATRIMONIO PROMEDIO'
    ],
    'VALOR': [4.5, 88.2, 1.2, 25.0, 8.5]
}
example_df = pd.DataFrame(example_data)
st.dataframe(example_df)


# --- 4. Subir archivo Excel ---
archivo = st.file_uploader("📁 Sube tu archivo de indicadores financieros (.xlsx)", type=["xlsx"])

df_processed = None
if archivo is not None:
    with st.spinner("Cargando y procesando archivo..."):
        try:
            df_raw = pd.read_excel(io.BytesIO(archivo.getvalue()))
            st.success("Archivo cargado exitosamente. Procesando datos...")

            df_raw['RUC'] = df_raw['RUC'].astype(str)

            df_filtered = df_raw[df_raw['INDICADOR_FINANCIERO'].isin(indicadores_elegidos_raw)].copy()

            # Rellenar valores nulos con la media global del indicador
            for ind in indicadores_elegidos_raw:
                mean_val_global = df_filtered[df_filtered['INDICADOR_FINANCIERO'] == ind]['VALOR'].mean()
                df_filtered.loc[df_filtered['INDICADOR_FINANCIERO'] == ind, 'VALOR'] = \
                    df_filtered.loc[df_filtered['INDICADOR_FINANCIERO'] == ind, 'VALOR'].fillna(mean_val_global)
            
            # Si aún hay NaN después de rellenar (ej. indicador completamente vacío)
            if df_filtered['VALOR'].isnull().any():
                st.warning("Se encontraron valores nulos persistentes después del preprocesamiento. Esto podría indicar indicadores completamente vacíos que se rellenarán con 0.")
                df_filtered['VALOR'] = df_filtered['VALOR'].fillna(0)


            # Pivotear la tabla para tener indicadores como columnas
            df_processed = df_filtered.pivot_table(index=['RUC', 'FECHA_CORTE'], columns='INDICADOR_FINANCIERO', values='VALOR').reset_index()
            df_processed.columns.name = None 

            # Renombrar columnas a los nombres cortos que espera el modelo
            df_processed = df_processed.rename(columns=indicadores_renamed)

            # Verificar que todas las columnas finales estén presentes y en orden
            missing_cols = [col for col in indicadores_model_order if col not in df_processed.columns]
            if missing_cols:
                st.error(f"❌ Error: Faltan las siguientes columnas de indicadores esperadas después de procesar tus datos: {', '.join(missing_cols)}."
                         " Asegúrate de que todos los 'INDICADOR_FINANCIERO' listados estén presentes en tu archivo y correctamente nombrados.")
                df_processed = None
            else:
                df_processed['FECHA_CORTE'] = pd.to_datetime(df_processed['FECHA_CORTE'])
                df_processed = df_processed[['RUC', 'FECHA_CORTE'] + indicadores_model_order].sort_values(['RUC', 'FECHA_CORTE'])
                st.write("Vista previa de los datos procesados (primeras 5 filas):")
                st.dataframe(df_processed.head())

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}. Por favor, verifica el formato de tu Excel y las columnas.")
            df_processed = None

if df_processed is not None:
    rucs_disponibles = df_processed['RUC'].unique()
    if len(rucs_disponibles) == 0:
        st.warning("No se encontraron RUCs válidos en el archivo procesado.")
    else:
        # Colocar el selectbox y el slider en una misma fila si es posible, o en columnas
        col1, col2 = st.columns([2, 1])
        with col1:
            ruc_input = st.selectbox("Selecciona el RUC de la cooperativa a predecir", rucs_disponibles)
        with col2:
            future_steps = st.slider("Número de meses a predecir", 1, 24, 6, key="future_steps_slider")

        if st.button("🚀 Generar Predicción"):
            st.subheader(f"Resultados de Predicción para RUC: {ruc_input}")

            df_ruc = df_processed[df_processed['RUC'] == ruc_input].sort_values('FECHA_CORTE')

            data = df_ruc[indicadores_model_order].values
            time_step = 12 # El modelo LSTM espera 12 pasos de tiempo

            if len(data) < time_step:
                st.warning(f"⚠️ No hay suficientes datos históricos ({len(data)} meses) para predecir para el RUC {ruc_input}. Se requieren al menos {time_step} meses de datos para una predicción robusta.")
            else:
                with st.spinner(f"Generando predicciones para {future_steps} meses..."):
                    data_scaled = scaler.transform(data)
                    last_sequence = data_scaled[-time_step:] # Últimos 'time_step' meses escalados
                    future_preds_original = [] 

                    for _ in range(future_steps):
                        input_seq = last_sequence.reshape(1, time_step, len(indicadores_model_order))
                        pred_scaled = model.predict(input_seq, verbose=0)
                        pred_original = scaler.inverse_transform(pred_scaled)[0]

                        if 'COBERTURA_CARTERA_PROBLEMATICA' in indicadores_model_order:
                            idx = indicadores_model_order.index('COBERTURA_CARTERA_PROBLEMATICA')
                            pred_original[idx] = max(0.0, pred_original[idx])

                        future_preds_original.append(pred_original)
                        last_sequence = np.append(last_sequence[1:], pred_scaled, axis=0)
                    st.success("✅ Predicciones generadas.")

                last_historic_date = df_ruc['FECHA_CORTE'].max()
                future_dates = pd.date_range(start=last_historic_date + pd.Timedelta(days=1),
                                             periods=future_steps, freq='M') 
                df_future = pd.DataFrame(future_preds_original, index=future_dates, columns=indicadores_model_order)

                st.subheader("📊 Predicciones para los próximos meses:")
                st.dataframe(df_future.style.format("{:.2f}")) 

                # --- 6. Graficar Resultados ---
                st.subheader("📈 Gráficos de Indicadores (Histórico vs. Predicción)")

                # Organizar los gráficos en filas y columnas
                num_indicators = len(indicadores_model_order)
                
                # Primera fila con 3 columnas
                cols_row1 = st.columns(3)
                # Segunda fila con 2 columnas (si hay suficientes indicadores)
                cols_row2 = st.columns(2) if num_indicators > 3 else []

                for i, col_name in enumerate(indicadores_model_order):
                    fig, ax = plt.subplots(figsize=(8, 4)) # Tamaño de figura para cada gráfico
                    ax.plot(df_ruc['FECHA_CORTE'], df_ruc[col_name], label='Histórico', color='blue', marker='o', markersize=4)
                    ax.plot(df_future.index, df_future[col_name], label='Predicción', color='red', linestyle='--', marker='x', markersize=4)
                    ax.axvline(x=df_ruc['FECHA_CORTE'].max(), color='green', linestyle=':', linewidth=2, label='Fin de Datos Históricos')
                    
                    # Usar el nombre "amigable" para el título del gráfico si existe en indicadores_renamed
                    # De lo contrario, usar el nombre de la columna.
                    friendly_name = next((raw_name for raw_name, short_name in indicadores_renamed.items() if short_name == col_name), col_name)
                    ax.set_title(f'{friendly_name}', fontsize=14)
                    ax.legend()
                    ax.grid(True, linestyle=':', alpha=0.7)
                    ax.tick_params(axis='x', rotation=45)
                    ax.set_ylabel('Valor')
                    plt.tight_layout()

                    # Asignar gráfico a la columna correcta
                    if i < 3:
                        with cols_row1[i]:
                            st.pyplot(fig)
                    else:
                        with cols_row2[i-3]: # Restar 3 para indexar correctamente en la segunda fila
                            st.pyplot(fig)
                    plt.close(fig) # Cerrar la figura para liberar memoria


else:
    st.info("Sube un archivo Excel para comenzar la predicción. Una vez cargado, podrás seleccionar un RUC y el número de meses a predecir.")
