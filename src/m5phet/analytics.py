"""INT11: an OPTIONAL embedded analytical projection over a run's local records.

This module is the only place DuckDB is imported, and nothing in the core path imports this module. The projection is a
rebuildable view of `metrics.jsonl` and `attempts.jsonl`: it is not a second authoritative accounting system, it starts no
server, and it never invents a value the records do not carry. Rebuilding it from the same records must give identical values,
units and populations.
"""

import json
import os
from pathlib import Path

PROJECTION_NAME = "analytics.duckdb"

METRIC_COLUMNS = ("run_id", "attempt_id", "metric", "definition", "definition_version", "unit", "scale",
                  "aggregation", "value", "task", "split", "horizon", "target", "population", "status")


def _require_duckdb():
    try:
        import duckdb
    except ImportError as exc:                                      # noqa: BLE001
        raise ImportError("the embedded analytics projection needs the optional 'duckdb' extra; core inference does not") from exc
    return duckdb


def build_projection(run_directory: Path, *, rebuild: bool = False) -> Path:
    """Materialise the projection beside the records. With `rebuild` the file is discarded and rebuilt from the same source."""
    duckdb = _require_duckdb()
    directory = Path(run_directory)
    target = directory / PROJECTION_NAME
    staging = directory / (PROJECTION_NAME + ".building")
    if staging.exists():
        staging.unlink()
    if target.exists() and not rebuild:
        return target
    rows = []
    metrics = directory / "metrics.jsonl"
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # the projection never decides what a broken record meant; recovery belongs to the evidence layer
                raise ValueError(f"{metrics.name} holds a record this projection cannot read; run EvidenceRun.recover() first")
            rows.append({column: record.get(column) for column in METRIC_COLUMNS})
    connection = duckdb.connect(str(staging))
    try:
        connection.execute("DROP TABLE IF EXISTS metrics")
        connection.execute(
            "CREATE TABLE metrics (run_id VARCHAR, attempt_id VARCHAR, metric VARCHAR, definition VARCHAR, "
            "definition_version VARCHAR, unit VARCHAR, scale VARCHAR, aggregation VARCHAR, value DOUBLE, task VARCHAR, "
            "split VARCHAR, horizon BIGINT, target VARCHAR, population BIGINT, status VARCHAR)")
        for row in rows:
            connection.execute("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               [row[c] for c in METRIC_COLUMNS])
        # the grain is preserved: a view that groups away split, target, definition or status would compare unlike rows, and
        # a sum of overlapping populations is a summed count, never a unique row coverage
        connection.execute(
            "CREATE OR REPLACE VIEW metrics_by_grain AS SELECT task, split, target, horizon, metric, definition, "
            "definition_version, unit, scale, aggregation, status, count(*) AS records, "
            "count(DISTINCT attempt_id) AS attempts, sum(population) AS summed_population, "
            "max(population) AS max_population FROM metrics "
            "GROUP BY 1,2,3,4,5,6,7,8,9,10,11")
    finally:
        connection.close()
    os.replace(staging, target)                                     # a rebuild is published atomically
    return target


def read_metrics(projection: Path) -> list:
    """Read the projection back as plain records, so a test can compare it with the source line by line."""
    duckdb = _require_duckdb()
    connection = duckdb.connect(str(projection))
    try:
        cursor = connection.execute(f"SELECT {', '.join(METRIC_COLUMNS)} FROM metrics ORDER BY horizon, metric")
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        connection.close()
