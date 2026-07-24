"""validate schema based on config logic"""

from typing import Dict, Any, Tuple, List
import polars as pl


class DataValidator:
    """
    validates dataframe against defined schema in config
    isolates invalid rows for review to avoid breaking pipeline
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _parse_validation_expressions(self) -> List[pl.Expr]:
        """convert config data checks into polars expressions for validation"""
        exprs = []
        for col_name, specs in self.config.get("columns", {}).items():
            check_specs = specs.get("checks", {})

            # valid range check
            if "range" in check_specs:
                r = check_specs["range"]
                exprs.append(pl.col(col_name).is_between(r["min"], r["max"]))

            # isin check (is value in list)
            if "isin" in check_specs:
                exprs.append(pl.col(col_name).is_in(check_specs["isin"]))

            # does text match regex pattern
            # patterns pulled fom similar checks in coworker's code
            if "regex" in check_specs:
                exprs.append(pl.col(col_name).str.contains(check_specs["regex"]))

            # check if null
            if not specs.get("nullable", False):
                exprs.append(pl.col(col_name).is_not_null())

        return exprs

    def validate_and_quarantine(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """splits input df into valid records and quarantined records."""
        validation_exprs = self._parse_validation_expressions()

        # combine all expressions with logical AND via all_horizontal
        is_valid_expr = pl.all_horizontal(validation_exprs)

        # DEBUG
        print(f" --- Schema Expressions --- ")
        print(is_valid_expr)

        clean_df = df.filter(is_valid_expr)

        # .not_() same as saying ~is_valid_expr (bitwise)
        quarantined_df = df.filter(is_valid_expr.not_())

        return clean_df, quarantined_df
