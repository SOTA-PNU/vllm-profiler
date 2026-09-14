"""CPU-only validation tests for derived product hash evidence."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from perfetto_hetero_profiler.support.files import sha256_file
from tools.evaluation.validation import (
    TrialValidationError,
    _valid_cross_source_join,
    _validate_derived_product_hashes,
)


class DerivedProductHashEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.roots = {
            "perfetto": root / "full",
            "focused": root / "focused",
            "overview": root / "overview",
        }
        files = {
            "perfetto": {
                "trace.pftrace": b"full trace",
                "trace.rbln-native.pftrace": b"native trace",
            },
            "focused": {
                "trace.pftrace": b"full trace",
                "trace.request-focused.pftrace": b"focused trace",
            },
            "overview": {
                "overview.json": b"{}\n",
                "overview.html": b"<html></html>\n",
            },
        }
        for root_name, entries in files.items():
            directory = self.roots[root_name]
            directory.mkdir()
            for name, content in entries.items():
                (directory / name).write_bytes(content)

    def evidence(self, *, legacy: bool = False) -> dict[str, object]:
        value: dict[str, object] = {
            "perfetto_byte_identical": True if legacy else None,
            "request_focused_perfetto_byte_identical": True if legacy else None,
            "overview_byte_identical": True if legacy else None,
            "perfetto_sha256": self.hashes("perfetto", "*.pftrace"),
            "request_focused_perfetto_sha256": self.hashes(
                "focused", "*.pftrace"
            ),
            "overview_sha256": self.hashes(
                "overview", "overview.json", "overview.html"
            ),
        }
        if not legacy:
            value["verification_mode"] = "single_production_generation"
        return value

    def hashes(self, root_name: str, *patterns: str) -> dict[str, str]:
        root = self.roots[root_name]
        paths = {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file()
        }
        return {path.name: sha256_file(path) for path in sorted(paths)}

    def assert_invalid(self, evidence: object) -> None:
        with self.assertRaises(TrialValidationError) as raised:
            _validate_derived_product_hashes(evidence, self.roots)
        self.assertEqual(
            str(raised.exception),
            "derived product hash evidence is invalid",
        )

    def test_single_generation_hashes_match_actual_products(self) -> None:
        _validate_derived_product_hashes(self.evidence(), self.roots)

    def test_combined_bundle_uses_only_canonical_full_and_focused_hashes(self) -> None:
        root = self.roots["perfetto"]
        (root / "trace.request-focused.pftrace").write_bytes(b"focused trace")
        roots = {**self.roots, "focused": root}
        evidence = self.evidence()
        evidence["perfetto_sha256"] = self.hashes(
            "perfetto", "trace.pftrace"
        )
        evidence["request_focused_perfetto_sha256"] = self.hashes(
            "perfetto", "trace.request-focused.pftrace"
        )
        _validate_derived_product_hashes(evidence, roots)

    def test_multi_request_bundle_does_not_require_focused_trace(self) -> None:
        root = self.roots["perfetto"]
        roots = {**self.roots, "focused": root}
        evidence = self.evidence()
        evidence["perfetto_sha256"] = self.hashes(
            "perfetto", "trace.pftrace"
        )
        evidence["request_focused_perfetto_sha256"] = {}
        _validate_derived_product_hashes(
            evidence,
            roots,
            require_request_focused=False,
        )

    def test_legacy_repeat_hashes_are_verified_and_absent_hashes_are_allowed(
        self,
    ) -> None:
        evidence = self.evidence(legacy=True)
        _validate_derived_product_hashes(evidence, self.roots)
        for field in (
            "perfetto_sha256",
            "request_focused_perfetto_sha256",
            "overview_sha256",
        ):
            evidence.pop(field)
        _validate_derived_product_hashes(evidence, self.roots)

    def test_altered_recorded_hash_is_rejected(self) -> None:
        evidence = self.evidence()
        hashes = evidence["perfetto_sha256"]
        assert isinstance(hashes, dict)
        digest = hashes["trace.pftrace"]
        hashes["trace.pftrace"] = ("0" if digest[0] != "0" else "1") + digest[1:]
        self.assert_invalid(evidence)

    def test_altered_file_content_is_rejected(self) -> None:
        evidence = self.evidence()
        (self.roots["overview"] / "overview.html").write_bytes(b"changed")
        self.assert_invalid(evidence)

    def test_missing_filename_is_rejected(self) -> None:
        evidence = self.evidence()
        hashes = evidence["request_focused_perfetto_sha256"]
        assert isinstance(hashes, dict)
        hashes.pop("trace.request-focused.pftrace")
        self.assert_invalid(evidence)

    def test_unknown_additional_filename_is_rejected(self) -> None:
        evidence = self.evidence()
        hashes = evidence["overview_sha256"]
        assert isinstance(hashes, dict)
        hashes["unexpected.json"] = "0" * 64
        self.assert_invalid(evidence)

    def test_malformed_or_unsafe_hash_entry_is_rejected(self) -> None:
        for field, name, value in (
            ("perfetto_sha256", "trace.pftrace", "A" * 64),
            ("perfetto_sha256", "../trace.pftrace", "0" * 64),
        ):
            with self.subTest(name=name):
                evidence = copy.deepcopy(self.evidence())
                hashes = evidence[field]
                assert isinstance(hashes, dict)
                hashes[name] = value
                self.assert_invalid(evidence)

    def test_empty_mapping_is_rejected(self) -> None:
        evidence = self.evidence()
        evidence["perfetto_sha256"] = {}
        self.assert_invalid(evidence)

    def test_single_generation_requires_explicit_null_repeat_fields(self) -> None:
        evidence = self.evidence()
        evidence.pop("overview_byte_identical")
        self.assert_invalid(evidence)


class CrossSourceJoinValidationTests(unittest.TestCase):
    @staticmethod
    def join(method: str) -> dict[str, object]:
        return {
            "status": "joined",
            "join_method": method,
            "missing_markers": [],
            "duplicate_markers": [],
            "ordering_violations": [],
            "pairing_issues": [],
        }

    def test_explicit_transfer_and_correlation_ids_are_valid(self) -> None:
        self.assertTrue(_valid_cross_source_join(self.join("transfer_id")))
        self.assertTrue(_valid_cross_source_join(self.join("correlation_id")))

    def test_fallback_or_inconsistent_join_is_rejected(self) -> None:
        self.assertFalse(_valid_cross_source_join(self.join("request_id")))
        value = self.join("transfer_id")
        value["ordering_violations"] = ["decode before transfer"]
        self.assertFalse(_valid_cross_source_join(value))


if __name__ == "__main__":
    unittest.main()
