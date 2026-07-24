"""
merge data and create merge ledger
FYI - I use merge and join interchangeably, polars uses join - need to change all instances
"""

from typing import Dict, Any, List
import polars as pl


class MergeLedger:
    """track merge metrics"""

    def __init__(self) -> None:
        # initialize merge history to log merge metrics
        self.merge_history: List[Dict[str, Any]] = []

    def merge_data(
        self,
        left_df: pl.DataFrame,
        right_df: pl.DataFrame,
        on: str,
        step_name: str,
        how: str = "left",  # default left merge
    ) -> pl.DataFrame:
        """executes polars joins and logs metrics"""
        left_shape = left_df.shape
        left_count = left_df.height

        right_shape = right_df.shape

        # execute primary join
        merged_df = left_df.join(right_df, on=on, how=how)

        # get unmatched left records via anti-join
        unmatched_left_df = left_df.join(right_df, on=on, how="anti")
        unmatched_count = unmatched_left_df.height

        ledger_entry = {
            "step_name": step_name,
            "join_type": how,
            "left_input_shape": left_shape,
            "right_input_shape": right_shape,
            "output_shape": merged_df.shape,
            "matched_both": left_count - unmatched_count,
            "left_only": unmatched_count,
            "unmatched_pct": (
                (unmatched_count / left_count * 100) if left_count > 0 else 0
            ),
        }

        self.merge_history.append(ledger_entry)

        return merged_df
