from fast_dataset_open import open_with_cache
import pandas as pd
import numpy as np


class DatasetLoader:
    def __init__(self,
                 alias,
                 params,
                 preload=False,
                 column_selector_x=None,
                 column_selector_y=None,
                 row_selector=None,
                 extra_columns_fn=lambda x: None):
        self.alias = alias
        self.params = params
        self.dataset_x = None
        self.dataset_y = None
        self.column_selector_x = column_selector_x
        self.column_selector_y = column_selector_y
        self.extra_colums = None
        self.extra_colums_fn = extra_columns_fn
        self.row_selector = row_selector

    def get(self):
        if self.dataset_x is not None:
            if self.extra_colums is not None:
                return self.dataset_x, self.dataset_y, self.extra_colums
            return self.dataset_x, self.dataset_y
        dataset = open_with_cache(*self.params)

        if self.row_selector is not None:
            selector = self.row_selector(dataset)
            dataset = dataset[selector]

        self.extra_colums = self.extra_colums_fn(dataset)

        if self.column_selector_x is not None:
            self.dataset_x = dataset[self.column_selector_x]
        else:
            self.dataset_x = dataset
        if self.column_selector_y is not None:
            self.dataset_y = dataset[self.column_selector_y]

        if self.extra_colums is not None:
            return self.dataset_x, self.dataset_y, self.extra_colums
        return self.dataset_x, self.dataset_y
