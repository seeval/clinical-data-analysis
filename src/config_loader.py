"""yaml config loader and polars data type mapping"""

import yaml
from typing import Dict, Any
import polars as pl

# map types from schema to polar types
POLARS_TYPE_MAP = {
    "String": pl.String,
    "Float64": pl.Float64,
    "Int64": pl.Int64,
    "Date": pl.Date,
    "Boolean": pl.Boolean,
}


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """reads yaml config from disk"""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def get_polars_schema(config: Dict[str, Any]) -> Dict[str, pl.DataType]:
    """convert string yaml types to polars types"""
    schema = {}

    for col_name, specs in config.get("columns", {}).items():
        dtype_spec = specs["dtype"]
        if dtype_spec in POLARS_TYPE_MAP:
            schema[col_name] = POLARS_TYPE_MAP[dtype_spec]
        else:
            raise ValueError(f"Unsupported polars data type: {dtype_spec}")

    return schema


if __name__ == "__main__":
    # testing load with print
    PATH = "config/schema_config.yml"
    cfg = load_yaml_config(PATH)
    print("RAW CONFIG")
    print("*" * 20)
    print(cfg)

    # check polars schema with print
    pl_schema = get_polars_schema(cfg)
    print("\nPL SCHEMA")
    print("*" * 20)
    print(pl_schema)
