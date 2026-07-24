# Clinical Data Processing Pipeline Example

A Polars-based data pipeline for standardizing, joining, validating, and reporting on multi-source clinical data.

## Overview

This project processes clinical data through a structured processing workflow:
1. **Standardization**: Standardizes column schemas, maps categorical variables, and handles missing or sentinel values (999.0).
2. **Merging**: Performs joins across procedure, readmission, and patient synthetic data sources.
3. **Validation**: Enforces target schema rules and quarantines non-compliant records.
4. **Reporting**: Generates descriptive summary table and visual distribution plots.

## Data Dictionary (`clean_df`)

| Column Name | Data Type | Source | Description |
| :--- | :--- | :--- | :--- |
| `procedure_id` | `String` | Source Raw | Unique ID, procedure id. |
| `patient_id` | `String` | Source Raw | Patient identifier. |
| `procedure_date` | `Date` | Source Raw | Date the surgical procedure was performed. |
| `joint_type` | `Categorical` | Mapped | Standardized joint strata (`HIP` or `KNEE`). |
| `age_at_procedure` | `Int64` | Calculated | Calculated age in years based on procedure date and patient birth date. |
| `bmi` | `Float64` | Standardized | Body Mass Index, out-of-range values and sentinels (`999.0`) imputed via strata median. |
| `surgical_duration_min` | `Float64` | Standardized | Total surgical time in minutes; invalid entries imputed via strata median. |
| `readmit_90day` | `Int64` | Calculated | Binary indicator (`1` = Readmitted within 90 days, `0` = No readmission). |

## Directory Structure

```text
.
├── config/
│   └── schema_config.yaml
├── src/
│   ├── config.py
│   ├── features.py
│   ├── merger.py
│   ├── reporter.py
│   ├── standardizer.py
│   └── validator.py
├── tests/
│   └── test_standardizer.py
├── main.py
├── pyproject.toml
```

## Setup & Installation

This project uses `uv` for dependency and environment management.

### Prerequisites

- Python 3.11+
- `uv` package manager

### Installation

1. Clone the repository and navigate to the project root:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies and build the virtual environment:
   ```bash
   uv sync
   ```

## Usage

### Run the Pipeline

Execute the primary pipeline orchestrator:

```bash
uv run python main.py
```

Generated summary table and visual figures will be written to the `reports/` directory.

### Run Unit Tests

Execute the test suite using `pytest`:

```bash
uv run python -m pytest tests/
```

### Alternative Setup (Standard `pip` / `venv`)

If you do not have `uv` installed, you can use standard Python tooling:

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

2. Install dependencies
    ```
   pip install -r requirements.txt

3. Run pipeline and tests
   ``` 
   python main.py
   pytest tests/

---

## Potential Enhancements

To scale this pipeline for production environments, the following minimum architecture upgrades are recommended:

### 1. Dedicated Logging & Observability
* Replace standard stdout `print()` statements with structured logging (e.g., Python `logging`).
* Implement log streams to capture runtime metadata,

### 2. Comprehensive Test Coverage
* **Edge Case Testing**: Expand `pytest` coverage to explicitly assert behavior against null-heavy data, empty DataFrames, duplicate primary keys, and unexpected data types.

### 3. Orchestration & Lineage Tracking
* Transition from hardcoded configuration files to environment-driven parameterization.
