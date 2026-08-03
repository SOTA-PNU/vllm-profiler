"""Tests for the installed rbln-smi 3.0.0 JSON contract."""

import json
from pathlib import Path
import subprocess
import unittest

from perfetto_hetero_profiler.collectors.npu import (
    RblnSmiClient,
    RblnSmiCommandError,
    RblnSmiParseError,
    parse_rbln_smi_json,
)
from perfetto_hetero_profiler.schema import Availability


FIXTURES = Path(__file__).parent / "fixtures" / "rbln_smi"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


class RblnSmiParserTests(unittest.TestCase):
    def test_normal_json_has_one_device(self):
        self.assertEqual(len(parse_rbln_smi_json(fixture("one_device.json")).rows), 1)

    def test_multiple_devices_preserve_order(self):
        rows = parse_rbln_smi_json(fixture("two_devices.json")).rows
        self.assertEqual([row.index for row in rows], [0, 1])

    def test_device_identity(self):
        row = parse_rbln_smi_json(fixture("one_device.json")).rows[0]
        self.assertEqual((row.index, row.name, row.device_id), (0, "RBLN-CA22", "npu-0"))

    def test_status_preserves_abnormal_value(self):
        row = parse_rbln_smi_json(fixture("two_devices.json")).rows[1]
        self.assertEqual(row.status, "warning")

    def test_kmd_and_firmware(self):
        result = parse_rbln_smi_json(fixture("one_device.json"))
        self.assertEqual(result.kmd_version, "3.0.0")
        self.assertEqual(result.rows[0].firmware_version, "3.0.0")

    def test_utilization_zero_is_available(self):
        value = parse_rbln_smi_json(fixture("one_device.json")).rows[0].utilization_percent
        self.assertEqual((value.availability, value.value), (Availability.AVAILABLE, 0.0))

    def test_memory_zero_is_available(self):
        value = parse_rbln_smi_json(fixture("one_device.json")).rows[0].memory_used_bytes
        self.assertEqual((value.availability, value.value), (Availability.AVAILABLE, 0))

    def test_power_zero_is_available(self):
        value = parse_rbln_smi_json(fixture("two_devices.json")).rows[0].power_watts
        self.assertEqual((value.availability, value.value), (Availability.AVAILABLE, 0.0))

    def test_memory_mib_converts_to_bytes(self):
        value = parse_rbln_smi_json(fixture("two_devices.json")).rows[1].memory_used_bytes
        self.assertEqual(value.value, 16 * 1024 * 1024)

    def test_memory_gib_converts_to_bytes(self):
        value = parse_rbln_smi_json(fixture("two_devices.json")).rows[1].memory_total_bytes
        self.assertEqual(value.value, 16 * 1024**3)

    def test_power_milliwatts_converts_to_watts(self):
        value = parse_rbln_smi_json(fixture("two_devices.json")).rows[1].power_watts
        self.assertAlmostEqual(value.value, 24.5)

    def test_power_microwatts_converts_to_watts(self):
        value = parse_rbln_smi_json(fixture("one_device.json")).rows[0].power_watts
        self.assertAlmostEqual(value.value, 25.0)

    def test_temperature_is_parsed_but_not_mapped_to_official_metric(self):
        value = parse_rbln_smi_json(fixture("one_device.json")).rows[0].temperature_celsius
        self.assertEqual(value.value, 53.0)

    def test_missing_memory_is_structurally_unsupported(self):
        value = parse_rbln_smi_json(
            fixture("unsupported_fields.json")
        ).rows[0].memory_used_bytes
        self.assertIs(value.availability, Availability.NOT_AVAILABLE)
        self.assertTrue(value.structurally_unsupported)
        self.assertIn("does not expose", value.reason)

    def test_missing_power_is_structurally_unsupported(self):
        value = parse_rbln_smi_json(
            fixture("unsupported_fields.json")
        ).rows[0].power_watts
        self.assertTrue(value.structurally_unsupported)

    def test_empty_util_is_unavailable_not_zero(self):
        value = parse_rbln_smi_json(
            fixture("unsupported_fields.json")
        ).rows[0].utilization_percent
        self.assertEqual((value.availability, value.value), (Availability.NOT_AVAILABLE, None))

    def test_null_temperature_is_unavailable(self):
        value = parse_rbln_smi_json(
            fixture("unsupported_fields.json")
        ).rows[0].temperature_celsius
        self.assertIs(value.availability, Availability.NOT_AVAILABLE)

    def test_malformed_metric_is_field_error_not_whole_document_failure(self):
        document = json.loads(fixture("two_devices.json"))
        document["devices"][1]["util"] = "broken"
        rows = parse_rbln_smi_json(json.dumps(document)).rows
        self.assertIs(rows[0].utilization_percent.availability, Availability.AVAILABLE)
        self.assertIs(rows[1].utilization_percent.availability, Availability.ERROR)

    def test_utilization_above_100_is_error(self):
        document = json.loads(fixture("one_device.json"))
        document["devices"][0]["util"] = "101"
        value = parse_rbln_smi_json(json.dumps(document)).rows[0].utilization_percent
        self.assertIs(value.availability, Availability.ERROR)

    def test_negative_power_is_error(self):
        document = json.loads(fixture("one_device.json"))
        document["devices"][0]["card_power"] = "-1uW"
        value = parse_rbln_smi_json(json.dumps(document)).rows[0].power_watts
        self.assertIs(value.availability, Availability.ERROR)

    def test_malformed_json_rejected(self):
        with self.assertRaises(RblnSmiParseError):
            parse_rbln_smi_json("{")

    def test_non_object_document_rejected(self):
        with self.assertRaisesRegex(RblnSmiParseError, "top-level"):
            parse_rbln_smi_json("[]")

    def test_empty_device_list_rejected(self):
        with self.assertRaisesRegex(RblnSmiParseError, "no NPU"):
            parse_rbln_smi_json('{"devices": []}')

    def test_bad_device_index_rejected(self):
        document = json.loads(fixture("one_device.json"))
        document["devices"][0]["npu"] = "bad"
        with self.assertRaisesRegex(RblnSmiParseError, "integer"):
            parse_rbln_smi_json(json.dumps(document))

    def test_duplicate_device_index_rejected(self):
        document = json.loads(fixture("two_devices.json"))
        document["devices"][1]["npu"] = 0
        with self.assertRaisesRegex(RblnSmiParseError, "unique"):
            parse_rbln_smi_json(json.dumps(document))

    def test_fixture_contains_no_real_identifiers(self):
        for path in FIXTURES.glob("*.json"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn('"sid"', text)
            self.assertNotIn('"uuid"', text)
            self.assertNotIn('"bus_id"', text)


class RblnSmiClientTests(unittest.TestCase):
    def test_default_argv_is_json(self):
        self.assertEqual(RblnSmiClient().argv, ("rbln-smi", "--json"))

    def test_selected_device_uses_documented_option(self):
        self.assertEqual(
            RblnSmiClient(device_ids=(0, 2)).argv,
            ("rbln-smi", "--json", "--device", "0,2"),
        )

    def test_negative_device_rejected(self):
        with self.assertRaises(ValueError):
            RblnSmiClient(device_ids=(-1,))

    def test_duplicate_device_rejected(self):
        with self.assertRaises(ValueError):
            RblnSmiClient(device_ids=(0, 0))

    def test_query_returns_parsed_result_and_raw_output(self):
        raw = fixture("one_device.json")
        client = RblnSmiClient(
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, raw, "")
        )
        result = client.query()
        self.assertEqual(result.raw_output, raw)

    def test_timeout_is_command_error(self):
        def runner(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 1)

        with self.assertRaisesRegex(RblnSmiCommandError, "timed out"):
            RblnSmiClient(runner=runner).query()

    def test_nonzero_exit_is_command_error(self):
        client = RblnSmiClient(
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 3, "", "device unavailable"
            )
        )
        with self.assertRaisesRegex(RblnSmiCommandError, "device unavailable"):
            client.query()

    def test_os_error_is_command_error(self):
        def runner(*args, **kwargs):
            raise FileNotFoundError("missing")

        with self.assertRaisesRegex(RblnSmiCommandError, "could not start"):
            RblnSmiClient(runner=runner).query()

    def test_parse_failure_is_command_error(self):
        client = RblnSmiClient(
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "{", "")
        )
        with self.assertRaisesRegex(RblnSmiCommandError, "parse error"):
            client.query()

    def test_version_strips_newline(self):
        client = RblnSmiClient(
            runner=lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 0, "3.0.0\n", ""
            )
        )
        self.assertEqual(client.version(), "3.0.0")

    def test_runner_never_uses_shell(self):
        captured = {}

        def runner(*args, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(args[0], 0, fixture("one_device.json"), "")

        RblnSmiClient(runner=runner).query()
        self.assertIs(captured["shell"], False)


if __name__ == "__main__":
    unittest.main()
