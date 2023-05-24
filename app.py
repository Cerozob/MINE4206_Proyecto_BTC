import streamlit as st
import ml_utils
import pandas as pd

st.write('# MINE4206 | forecasting del valor de cierre para BTCUSDT')

arima_available_models = ml_utils.get_available_arima_models()
nn_available_models = ml_utils.get_available_rnn_models()

global_results_dataframe = pd.DataFrame()

arima_options = st.multiselect(
    'Elige qué modelos ARIMA quieres usar',
    arima_available_models,
    format_func=lambda x: x.stem,
    default=[],
    help='Selecciona los modelos que quieras usar'
)

rnn_options = st.multiselect(
    'Elige qué modelos LSTM quieres usar',
    nn_available_models,
    format_func=lambda x: x.stem,
    default=[],
    help='Selecciona los modelos que quieras usar'
)


def call_prediction_arima(models=[]):
    return ml_utils.predict_with_arimas(models) if models else None


def call_prediction_rnn(models=[]):
    return ml_utils.predict_with_rnns(models) if models else None


if arima_options or rnn_options:
    st.write(
        f'Has elegido los siguientes {len(arima_options+rnn_options)} modelos:')
    st.write(arima_options+rnn_options)
    all_results = []
    if st.button('**Predecir**'):
        with st.spinner(f'Prediciendo resultados con {len(arima_options)} ARIMAs...'):
            arimaresults = call_prediction_arima(arima_options)
        if arimaresults is None or len(arimaresults) == 0:
            st.write('No hay modelos ARIMA seleccionados')
        else:
            st.write('Resultados ARIMA')
            for model, result in arimaresults:
                st.write(model)
                st.dataframe(result)
                all_results.append(result)
        with st.spinner(f'Prediciendo resultados con {len(rnn_options)} LSTMs...'):
            rnnresults = call_prediction_rnn(rnn_options)
        if rnnresults is None or len(rnnresults) == 0:
            st.write('No hay modelos LSTM seleccionados')
        else:
            st.write('Resultados LSTM')
            for model, result in rnnresults:
                st.write(model)
                st.dataframe(result)
                all_results.append(result)

if st.button('Limpiar resultados'):
    global_results_dataframe = pd.DataFrame()
