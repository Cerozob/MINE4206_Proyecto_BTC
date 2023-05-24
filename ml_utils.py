import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow import stack
from keras.utils import timeseries_dataset_from_array
from keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import datetime


TRAIN_MEAN = 19800.415727  # del notebook
TRAIN_STD = 16245.150479  # del notebook

DATAPATH = Path('./data')
MODELSPATH = Path('./models')
ARIMAPATH = MODELSPATH / 'arima'
LSTMPATH = MODELSPATH / 'lstm'


def load_data(path=Path("./data/data.parquet")):
    dframe = pd.read_parquet(path)
    dframe.drop(["open", "high", "low"], axis=1, inplace=True)
    close_time = dframe.pop("close_time")
    # re put close_time
    # dframe.insert(0, "close_time", close_time)
    return dframe, close_time


def normalize_dataset(df, train_mean=TRAIN_MEAN, train_std=TRAIN_STD):
    return (df - train_mean)/train_std


def denormalize_dataset(df, train_mean=TRAIN_MEAN, train_std=TRAIN_STD):
    return (df * train_std) + train_mean


DATA, CLOSE_TIME = load_data()
NORM_DATA = normalize_dataset(DATA)


def load_arima_model(name="arima.pkl"):
    arimapath = ARIMAPATH / name
    arima_model = ARIMAResults.load(arimapath)
    return arima_model


def load_sarimax_model(name="sarimax.pkl"):
    path = ARIMAPATH / name
    sarimax_model = SARIMAXResults.load(path)
    return sarimax_model

# ! LSTM stuff


def load_rnn_model(name="LSTM_normalized.h5"):
    path = LSTMPATH / name
    rnn_model = load_model(path)
    return rnn_model


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df=None, test_df=None,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


LSTM_TIMEUNITS_PAST = 24*7*4
LSTM_TIMEUNITS_FUTURE = 24*7
LSTM_TIMEUNITS_SHIFT = 24

LSTM_OUTPUT_LABELS = list(DATA.columns)
LSTM_N_OUTPUTS = len(LSTM_OUTPUT_LABELS)
LSTM_MULTI_WINDOW = WindowGenerator(input_width=LSTM_TIMEUNITS_PAST,
                                    label_width=LSTM_TIMEUNITS_FUTURE,
                                    shift=LSTM_TIMEUNITS_SHIFT, label_columns=LSTM_OUTPUT_LABELS, train_df=NORM_DATA)


def get_window(hours_past=24*7*4, normalized_data=NORM_DATA):
    # gets the last 24*7*4 hours of data from the normalized dataset
    # 32 es el batch size del lstm
    selection = normalized_data[-hours_past*32:]
    # 32 es el batch size, si no el lstm no funciona
    times = normalized_data[-hours_past*32:]
    return times, selection


def gen_dates(pfrom=CLOSE_TIME.iloc[-1], to=None, dframe=None):
    to = pfrom+datetime.timedelta(hours=dframe.shape[0]) if to is None else to
    return pd.Series(pd.date_range(start=pfrom, end=to, freq="1H", inclusive="right"))


def get_past_norm_dframe(pfrom=LSTM_TIMEUNITS_PAST):
    past_data = NORM_DATA.iloc[-pfrom:]
    past_dates = CLOSE_TIME.iloc[-pfrom:]
    past_data.insert(0, "close_time", past_dates)
    # past_data.insert(0, "is_prediction", False)
    past_data.set_index("close_time", inplace=True)
    past_data.reset_index(inplace=True)
    return past_data


def get_past_dframe(pfrom=LSTM_TIMEUNITS_PAST):
    past_data = DATA.iloc[-pfrom:]
    past_dates = CLOSE_TIME.iloc[-pfrom:]
    past_data.insert(0, "close_time", past_dates)
    # past_data.insert(0, "is_prediction", False)
    past_data.set_index("close_time", inplace=True)
    past_data.reset_index(inplace=True)
    return past_data


def get_all_past_dframe():
    past_data = DATA.iloc[:]
    past_dates = CLOSE_TIME.iloc[:]
    past_data.insert(0, "close_time", past_dates)
    # past_data.insert(0, "is_prediction", False)
    past_data.set_index("close_time", inplace=True)
    past_data.reset_index(inplace=True)
    return past_data


def lstm_predict(model=None, window=LSTM_MULTI_WINDOW):
    model = load_rnn_model() if model is None else model
    latest_data = get_window()
    latest_dataset = window.make_dataset(latest_data[1])
    result = model.predict(latest_dataset)
    results_dframe = pd.DataFrame(data=result[-1], columns=LSTM_OUTPUT_LABELS)
    last_time = CLOSE_TIME.iloc[-1]
    next_time = last_time + pd.Timedelta(hours=results_dframe.shape[0])
    past_dates_used = CLOSE_TIME.iloc[-LSTM_TIMEUNITS_PAST:]
    future_dates = gen_dates(pfrom=last_time, dframe=results_dframe)
    # put everything together
    results_dframe.insert(0, "close_time", future_dates)
    # results_dframe.insert(0, "is_prediction", True)
    results_dframe.set_index("close_time", inplace=True)
    results_dframe.reset_index(inplace=True)
    return results_dframe, past_dates_used, next_time


def lstm_stitch_predictions_norm(past_dframe=None, predictions_dframe=None
                                 ):
    """
    past_dframe: dataframe with the past data
    predictions_dframe: dataframe with the predictions

    returns: dataframe with the past and the predictions

    this functions supposes that both dataframes are normalized, so it denormalizes them

    """
    # stitch past and predictions
    past_dframe = get_past_norm_dframe() if past_dframe is None else past_dframe
    predictions_dframe = lstm_predict(
    )[0] if predictions_dframe is None else predictions_dframe
    past_dframe = pd.concat([past_dframe, predictions_dframe])
    past_dframe.set_index("close_time", inplace=True)
    past_dframe.sort_index(inplace=True)
    past_dframe.reset_index(inplace=True)
    # denormalize
    past_dframe[LSTM_OUTPUT_LABELS] = denormalize_dataset(
        past_dframe[LSTM_OUTPUT_LABELS])
    return past_dframe

# ! ARIMA & SARIMA stuff


ARIMA_TIMEUNITS_FUTURE = 7  # los arima estan en dias


def arima_get_past_dframe(pfrom=LSTM_TIMEUNITS_PAST):
    past_data = DATA.iloc[-pfrom:]
    past_dates = CLOSE_TIME.iloc[-pfrom:]
    past_data.insert(0, "close_time", past_dates)
    past_data = past_data.resample(
        "1D", on="close_time").mean(numeric_only=False)
    # past_data.insert(0, "is_prediction", False)
    past_data.reset_index(inplace=True)
    past_data.set_index("close_time", inplace=True)
    # de verdad se hace ambas veces a proposito
    past_data.reset_index(inplace=True)
    past_data.reset_index(inplace=True)
    return past_data


def arima_predict(model=None):
    model = load_arima_model() if model is None else model
    preds = model.forecast(steps=ARIMA_TIMEUNITS_FUTURE).to_frame()
    preds = preds.resample("1H").ffill()
    preds.rename(columns={'predicted_mean': 'close'}, inplace=True)
    preds.index.name = "close_time"
    preds.reset_index(inplace=True)
    # preds.insert(0, "is_prediction", True)
    return preds


def arima_stitch_predictions(past_dframe=None, predictions_dframe=None):
    """
    past_dframe: dataframe with the past data not normalized
    predictions_dframe: dataframe with the predictions not normalized

    returns: dataframe with the past and the predictions

    this functions supposes that neither dataframes are normalized, so it doesnt denormalize them

    """
    # stitch past and predictions
    past_dframe = get_past_dframe() if past_dframe is None else past_dframe
    predictions_dframe = arima_predict(
    ) if predictions_dframe is None else predictions_dframe
    past_dframe = pd.concat([past_dframe, predictions_dframe])
    past_dframe.set_index("close_time", inplace=True)
    past_dframe.sort_index(inplace=True)
    past_dframe.reset_index(inplace=True)
    # denormalize
    return past_dframe


def get_available_arima_models():
    models = []
    for model in ARIMAPATH.iterdir():
        models.append(model)
    return models


def get_available_rnn_models():
    models = []
    for model in LSTMPATH.iterdir():
        models.append(model)
    return models


def list_all_models():
    models = []
    for model in ARIMAPATH.iterdir():
        models.append(model)
    for model in LSTMPATH.iterdir():
        models.append(model)
    for model in Path("./models/unused").iterdir():
        models.append(model)
    return models


def predict_with_arimas(models=[]):
    if len(models) == 0:
        return []
    results = []
    for model in models:
        if "sarima" in model.stem:
            arimamodel = load_sarimax_model(model.name)
        else:
            arimamodel = load_arima_model(model.name)
        preds = arima_predict(arimamodel)
        results.append((model.stem, preds))
    return results


def predict_with_rnns(models=[]):
    if len(models) == 0:
        return []
    results = []
    for model in models:
        rnnmodel = load_rnn_model(model.name)
        preds = lstm_predict(rnnmodel)[0]
        preds[["close"]] = denormalize_dataset(preds[["close"]])
        results.append((model.stem, preds))
    return results


if __name__ == "__main__":
    pass
