from typing import List,Optional
import pandas as pd

def validate_dataframe(pd_df: pd.DataFrame,
                    cols: List[str],
                    sort_by: Optional[str] = None):
    _columns = cols.copy()
    if sort_by and sort_by not in _columns:
        _columns.append(sort_by)
    for col in _columns:
        if col not in pd_df.columns:
            raise AttributeError(f"{col} is not a DataFrame's column")