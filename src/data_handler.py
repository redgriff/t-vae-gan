import decimal
import uuid
from typing import List

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer


class DataHandler:
    def __init__(self, df: pd.DataFrame):
        self.org_df: pd.DataFrame = df

        self._cat_scaler = None
        self._cat_columns = None
        self._cat_scaled_np = None

        self._num_scaler = None
        self._num_columns = None
        self._num_scaled_np = None

        self.none_guid = str(uuid.uuid4())

    def transform(self):
        self._cat_columns, self._num_columns = self._get_cat_num(self.org_df)

        self._cat_scaler, self._cat_scaled_np = self._scale_categoricals(self.org_df[self._cat_columns])
        self._num_scaler, self._num_scaled_np, = self._scale_numericals(self.org_df[self._num_columns])

        if self._cat_scaled_np is not None and self._num_scaled_np is not None:
            scaled_np = np.concatenate([self._cat_scaled_np, self._num_scaled_np], axis=1)
        else:
            scaled_np = self._cat_scaled_np if self._cat_scaled_np is not None else self._num_scaled_np

        return scaled_np

    def _scale_categoricals(self, cat_df):
        cat_scaled_np = None
        cat_scaler = None

        if len(cat_df.columns) > 0:
            if len(cat_df.isnull().any(axis=1)) > 0:
                cat_df = cat_df.fillna(self.none_guid)

            cat_scaler = {"onehot": {}, "minmax": MinMaxScaler((-1, 1), clip=True)}

            lst = []
            for column in cat_df.columns:
                cat_scaler["onehot"][column] = LabelBinarizer()
                cat_scaler["onehot"][column].fit(cat_df[column])
                lst.append(cat_scaler["onehot"][column].transform(cat_df[column]))

            cat_scaled_np = np.concatenate(lst, axis=1)

            cat_scaler["minmax"].fit(cat_scaled_np)
            cat_scaled_np = cat_scaler["minmax"].transform(cat_scaled_np)

        return cat_scaler, cat_scaled_np

    def _scale_numericals(self, num_df):
        num_scaler = None
        num_scaled_np = None

        if len(num_df.columns) > 0:
            num_scaler = MinMaxScaler((-1, 1), clip=True)
            num_scaler.fit(num_df)
            num_scaled_np = num_scaler.transform(num_df)

        return num_scaler, num_scaled_np

    def get_cat_sizes(self):
        cat_sizes = []
        if len(self._cat_columns) > 0:
            for column in self._cat_scaler["onehot"]:
                size = len(self._cat_scaler["onehot"][column].classes_)
                cat_sizes.append(size if size > 2 else 1)

        return cat_sizes

    def get_num_sizes(self):
        num_sizes = []
        if len(self._num_columns) > 0:
            for num_column in self._num_columns:
                num_sizes.append(len(self.org_df[num_column].unique()))

        return num_sizes

    def reverse(self, scaled_np: np.ndarray):
        cat_scaled_np, num_scaled_np, _ = np.split(scaled_np, [sum(self.get_cat_sizes()), sum(self.get_cat_sizes()) + len(self.get_num_sizes())], axis=1)

        cat_rev_df = None
        if len(self.get_cat_sizes()) > 0:
            cat_rev_np = self._cat_scaler["minmax"].inverse_transform(cat_scaled_np)

            cat_sizes = self.get_cat_sizes()
            indices_or_sections = [cat_sizes[0]]
            indice = cat_sizes[0]
            for i in range(1, len(cat_sizes)):
                indices_or_sections.append(cat_sizes[i]+indice)
                indice = cat_sizes[i]+indice
            cat_columns_np = np.split(cat_rev_np, indices_or_sections, axis=1)
            cat_columns_np = cat_columns_np[:-1]

            lst = []
            for column_name, cat_column_np in zip(self._cat_scaler["onehot"], cat_columns_np):
                lst.append(self._cat_scaler["onehot"][column_name].inverse_transform(cat_column_np))
            cat_columns_rev_np = np.stack(lst, axis=1)
            cat_rev_df = pd.DataFrame(cat_columns_rev_np, columns=self._cat_columns)
            cat_rev_df = cat_rev_df.replace(self.none_guid, np.nan)

        num_rev_df = None
        if len(self.get_num_sizes()) > 0:
            num_rev_df = pd.DataFrame(self._num_scaler.inverse_transform(num_scaled_np), columns=self._num_columns)

        rev_df = pd.concat(list(filter(None.__ne__, [cat_rev_df, num_rev_df])), axis=1)

        rev_df = rev_df[self.org_df.columns]
        for column, org_dtype in zip(self.org_df.columns, self.org_df.dtypes):
            if org_dtype == np.dtype("float"):
                decimal_digit = self._get_max_decimal_digit(list(self.org_df[column]))
                rev_df[column] = rev_df[column].apply(lambda x: round(x, decimal_digit))
            if org_dtype == np.dtype("int64"):
                rev_df[column] = rev_df[column].apply(lambda x: round(x))
            rev_df[column] = rev_df[column].astype(org_dtype)

        return rev_df

    def _get_max_decimal_digit(self, numbers: List[str]):

        def _get_decimal_digit(number: str):
            d = decimal.Decimal(number)
            return d.as_tuple().exponent * (-1)

        decimal_digits = [_get_decimal_digit(str(number))for number in numbers]
        max_decimal_digit = max(decimal_digits)
        return max_decimal_digit

    @staticmethod
    def _get_cat_num(df: pd.DataFrame):
        cat_columns = []
        num_columns = []

        for column, dtype in zip(df.columns, df.dtypes):
            unique_values = len(df[column].unique())
            if dtype in [np.dtype("object"), np.dtype("bool")]:
                cat_columns.append(column)
                print(f"columns:{column}, type:{dtype}, added: categorical, unique: {unique_values}")
            else:
                num_columns.append(column)
                print(f"columns:{column}, type:{dtype}, added: numerical, unique: {unique_values}")

        return cat_columns, num_columns
