"""CPU-only contracts for official Nsight SQLite export schemas."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import re
import sqlite3
import unittest
from unittest import mock

from perfetto_hetero_profiler.perfetto.native_details import (
    NativeDetailError,
    _ClockBridge,
    _attach_explicit_flows,
)
from perfetto_hetero_profiler.perfetto.native_nsys import (
    SUPPORTED_NSYS_EXPORT_SCHEMA_VERSIONS,
    _read_nsys_export_schema_version,
    _read_nsys_rows,
    _validate_nsys_sqlite_preamble,
)


SCHEMA_VERSION = "3.16.1"


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _metadata_connection(
    *,
    rows: tuple[object, ...] = (SCHEMA_VERSION,),
    definition: str = "name TEXT NOT NULL, value TEXT",
) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(f"CREATE TABLE META_DATA_EXPORT ({definition})")
    if {"name", "value"}.issubset(
        {row[1] for row in connection.execute("PRAGMA table_info(META_DATA_EXPORT)")}
    ):
        connection.executemany(
            "INSERT INTO META_DATA_EXPORT(name, value) VALUES (?, ?)",
            (("EXPORT_SCHEMA_VERSION", value) for value in rows),
        )
    return connection


def _event_connection() -> sqlite3.Connection:
    connection = _metadata_connection()
    connection.executescript(
        """
        CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER);
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER, name TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, eventClass INTEGER,
            globalTid INTEGER, correlationId INTEGER, nameId INTEGER,
            returnValue INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER,
            contextId INTEGER, streamId INTEGER, correlationId INTEGER,
            globalPid INTEGER, demangledName INTEGER, shortName INTEGER,
            gridX INTEGER, gridY INTEGER, gridZ INTEGER,
            blockX INTEGER, blockY INTEGER, blockZ INTEGER,
            registersPerThread INTEGER, staticSharedMemory INTEGER,
            dynamicSharedMemory INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, deviceId INTEGER,
            contextId INTEGER, streamId INTEGER, correlationId INTEGER,
            globalPid INTEGER, bytes INTEGER, copyKind INTEGER,
            srcKind INTEGER, dstKind INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
            start INTEGER, end INTEGER, deviceId INTEGER,
            contextId INTEGER, streamId INTEGER, correlationId INTEGER,
            globalPid INTEGER, value INTEGER, bytes INTEGER,
            memKind INTEGER
        );
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, label TEXT);
        CREATE TABLE ENUM_CUDA_MEM_KIND (id INTEGER, label TEXT);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, eventType INTEGER, rangeId INTEGER,
            text TEXT, globalTid INTEGER, textId INTEGER, domainId INTEGER
        );
        INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (100);
        INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel');
        INSERT INTO StringIds VALUES (2, 'syntheticKernel');
        INSERT INTO PROCESSES
            VALUES (328326543572992, 1, 'synthetic-process');
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME
            VALUES (10, 20, 0, 328326546365563, 7, 1, 0);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
            VALUES (
                30, 40, 0, 1, 2, 7, 328326543572992, 2, 2,
                1, 1, 1, 32, 1, 1, 16, 0, 0
            );
        """
    )
    return connection


class NsightSchemaVersionTests(unittest.TestCase):
    def test_supported_version_is_explicit_and_accepted(self) -> None:
        self.assertEqual(SUPPORTED_NSYS_EXPORT_SCHEMA_VERSIONS, (SCHEMA_VERSION,))
        with _metadata_connection() as connection:
            self.assertEqual(
                _read_nsys_export_schema_version(
                    connection,
                    tables=_tables(connection),
                ),
                SCHEMA_VERSION,
            )

    def test_metadata_table_is_required(self) -> None:
        with sqlite3.connect(":memory:") as connection:
            with self.assertRaisesRegex(
                NativeDetailError,
                r"META_DATA_EXPORT table is missing.*observed=.*<missing>.*3\.16\.1",
            ):
                _read_nsys_export_schema_version(connection, tables=set())

    def test_name_and_value_fields_are_required(self) -> None:
        for definition, missing in (
            ("key TEXT, value TEXT", "name"),
            ("name TEXT", "value"),
        ):
            with self.subTest(missing=missing), _metadata_connection(
                rows=(), definition=definition
            ) as connection:
                with self.assertRaisesRegex(
                    NativeDetailError,
                    rf"field is missing: \['{missing}'\].*supported=.*3\.16\.1",
                ):
                    _read_nsys_export_schema_version(
                        connection,
                        tables=_tables(connection),
                    )

    def test_version_row_and_nonempty_value_are_required(self) -> None:
        with _metadata_connection(rows=()) as connection:
            with self.assertRaisesRegex(
                NativeDetailError,
                r"row is missing.*observed=.*<missing>",
            ):
                _read_nsys_export_schema_version(
                    connection,
                    tables=_tables(connection),
                )
        for value in ("", "   "):
            with self.subTest(value=value), _metadata_connection(
                rows=(value,)
            ) as connection:
                with self.assertRaisesRegex(
                    NativeDetailError,
                    r"is empty.*supported=.*3\.16\.1",
                ):
                    _read_nsys_export_schema_version(
                        connection,
                        tables=_tables(connection),
                    )

    def test_non_text_and_malformed_versions_are_rejected(self) -> None:
        with _metadata_connection(
            rows=(sqlite3.Binary(b"3.16.1"),)
        ) as connection:
            with self.assertRaisesRegex(
                NativeDetailError,
                r"non-text.*BLOB length=6.*supported=.*3\.16\.1",
            ):
                _read_nsys_export_schema_version(
                    connection,
                    tables=_tables(connection),
                )
        for value in ("3.16", "3.16.1.0", "03.16.1", "3.16.x"):
            with self.subTest(value=value), _metadata_connection(
                rows=(value,)
            ) as connection:
                with self.assertRaisesRegex(
                    NativeDetailError,
                    rf"format is invalid.*{re.escape(value)}.*3\.16\.1",
                ):
                    _read_nsys_export_schema_version(
                        connection,
                        tables=_tables(connection),
                    )

    def test_future_version_fails_closed_with_observed_and_supported(self) -> None:
        with _metadata_connection(rows=("3.17.0",)) as connection:
            with self.assertRaisesRegex(
                NativeDetailError,
                r"unsupported.*observed=\['3\.17\.0'\].*supported=\['3\.16\.1'\]",
            ) as error:
                _read_nsys_export_schema_version(
                    connection,
                    tables=_tables(connection),
                )
        self.assertNotIn("/home/", str(error.exception))

    def test_duplicate_and_conflicting_versions_are_rejected(self) -> None:
        for values in (
            (SCHEMA_VERSION, SCHEMA_VERSION),
            (SCHEMA_VERSION, "3.17.0"),
        ):
            with self.subTest(values=values), _metadata_connection(
                rows=values
            ) as connection:
                with self.assertRaisesRegex(
                    NativeDetailError,
                    r"duplicated or conflicting.*observed=.*3\.16\.1",
                ):
                    _read_nsys_export_schema_version(
                        connection,
                        tables=_tables(connection),
                    )


class NsightPreambleAndEventTests(unittest.TestCase):
    def test_quick_check_failure_precedes_all_table_queries(self) -> None:
        connection = mock.Mock()
        cursor = mock.Mock()
        cursor.fetchall.return_value = [("database disk image is malformed",)]
        connection.execute.return_value = cursor
        with self.assertRaisesRegex(NativeDetailError, "quick_check failed"):
            _validate_nsys_sqlite_preamble(connection)
        connection.execute.assert_called_once_with("PRAGMA quick_check")

    def test_schema_version_precedes_existing_required_table_validation(self) -> None:
        with _metadata_connection(rows=("3.17.0",)) as connection:
            with self.assertRaisesRegex(NativeDetailError, "unsupported"):
                _validate_nsys_sqlite_preamble(connection)
        with _metadata_connection() as connection:
            with self.assertRaisesRegex(
                NativeDetailError,
                "lacks required tables",
            ):
                _validate_nsys_sqlite_preamble(connection)

    def test_supported_schema_preserves_nsight_slices_and_correlation_flow(self) -> None:
        bridge = _ClockBridge(
            source_role="gpu",
            native_clock_domain="nsys-native",
            native_timestamp_unit="nsight-report-native",
            offset_ns=0,
            observed_half_range_ns=0,
            uncertainty_ns=1,
            canonical_offset_ns=0,
            sample_offsets_ns=(0,),
        )
        with _event_connection() as connection:
            self.assertEqual(
                _validate_nsys_sqlite_preamble(connection),
                SCHEMA_VERSION,
            )
            slices, _, counts, metadata_count = _read_nsys_rows(
                SimpleNamespace(manifest=SimpleNamespace(run_id="run")),
                connection,
                strings={1: "cudaLaunchKernel", 2: "syntheticKernel"},
                process_names={
                    328_326_543_572_992: (1, "synthetic-process")
                },
                session_unix_ns=100,
                bridge=bridge,
            )
        converted, flows = _attach_explicit_flows("run", "gpu_nsys", slices)
        self.assertEqual(
            counts,
            Counter({"CUDA Runtime API": 1, "CUDA kernels": 1}),
        )
        self.assertEqual(metadata_count, 0)
        self.assertEqual(
            [(item.spec.name, item.spec.timestamp_ns) for item in converted],
            [("cudaLaunchKernel", 110), ("syntheticKernel", 130)],
        )
        self.assertEqual(len(flows), 1)
        self.assertEqual(
            flows[0].correlation_id,
            "gpu_nsys:nsight-process:328326543572992:7",
        )
if __name__ == "__main__":
    unittest.main()
