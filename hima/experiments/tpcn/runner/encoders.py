import numpy as np
from typing import Union, List
from numpy.typing import NDArray

    
class SimpleOneHotEncoder:
    def __init__(self, max_categories: int):
        
        self.max_categories = max_categories
        self.categories = {}
    
    def fit(self, x: Union[NDArray[Union[np.int32, np.int16]], List[int]], int) -> None:
        
        if isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, int):
            x = np.array([x])
        
        unique_values = np.unique(x)

        for value in unique_values:
            self.categories[int(value)] = len(self.categories)
        
        return self

    def transform(self, x: Union[NDArray[Union[np.int32, np.int16]], List[int]]) -> NDArray:
        
        if isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, int):
            x = np.array([x])
        
        n_categories = np.unique(x).shape[0]
    
        if n_categories > self.max_categories:
            raise RuntimeError(f'The number of unique observations' \
                            f'{n_categories} is greater than the maximum allowed {self.max_categories}')

        result = np.zeros((x.shape[0], self.max_categories), dtype=np.float32)

        for i, value in enumerate(x):

            if int(value) not in self.categories.keys():
                self.categories[int(value)] = len(self.categories)
            
            result[i, self.categories[value]] = 1.0
        
        return result