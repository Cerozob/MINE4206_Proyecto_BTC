import streamlit as st
import ml_utils
import pandas as pd

st.write('# MINE4206 | forecasting del valor de cierre para BTCUSDT')

arima_available_models = ml_utils.get_available_arima_models()
nn_available_models = ml_utils.get_available_rnn_models()

global_results_dataframe = ml_utils.get_past_dframe()

names = []

arima_options = st.multiselect(
    'Elegir modelos ARIMA a usar',
    arima_available_models,
    format_func=lambda x: x.stem.upper(),
    default=[],
    help='Selecciona los modelos ARIMA o SARIMA que quieras usar'
)

rnn_options = st.multiselect(
    'Elegir modelos RNN a usar',
    nn_available_models,
    format_func=lambda x: x.stem.upper(),
    default=[],
    help='Selecciona los modelos LSTM que quieras usar'
)


def call_prediction_arima(models=[]):
    return ml_utils.predict_with_arimas(models) if models else None


def call_prediction_rnn(models=[]):
    return ml_utils.predict_with_rnns(models) if models else None


if arima_options or rnn_options:
    st.write(
        f'Has elegido los siguientes {len(arima_options+rnn_options)} modelos:')
    st.write([m.stem for m in (arima_options+rnn_options)])

    if st.button('**Predecir**'):
        if len(rnn_options) > 0:
            lstmwarn = st.warning(
                "Los modelos LSTM pueden tardar bastante más tiempo en predecir, pero mostraron ser más precisos en nuestras pruebas", icon="⚠️")
        with st.spinner(f'Prediciendo resultados con {len(arima_options)} ARIMAs y {len(rnn_options)} LSTMs...'):
            arimaresults = call_prediction_arima(arima_options)
            if arimaresults is None or len(arimaresults) == 0:
                st.warning('No hay modelos ARIMA seleccionados', icon="⚠️")
            else:
                # st.write('Resultados ARIMA')
                for name, result in arimaresults:
                    # st.write(model)
                    # st.dataframe(result)
                    names.append(name)
                    result.rename(
                        columns={'close': name}, inplace=True)
                    # add a column with the name of the model, and add the prediction rows in the respective date
                    global_results_dataframe = pd.merge(
                        global_results_dataframe, result, how='outer', on='close_time')
            # with st.spinner(f'Prediciendo resultados con {len(rnn_options)} LSTMs...'):
            rnnresults = call_prediction_rnn(rnn_options)
            if rnnresults is None or len(rnnresults) == 0:
                lstmwarn = st.warning('No hay modelos LSTM seleccionados', icon="⚠️")
            else:
                lstmwarn.empty()
                for name, result in rnnresults:
                    # st.write(model)
                    # st.dataframe(result)
                    names.append(name)
                    result.rename(
                        columns={'close': name}, inplace=True)
                    global_results_dataframe = pd.merge(
                        global_results_dataframe, result, how='outer', on='close_time')
else:
    st.error('Selecciona al menos un modelo para ver resultados de predicción', icon="🪅")

thechart = st.line_chart(global_results_dataframe, use_container_width=True,
                         x="close_time", y=["close"]+names)
# st.dataframe(global_results_dataframe)
