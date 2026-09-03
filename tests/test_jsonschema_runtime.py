"""Runtime JSON Schema structure and parity-corpus tests."""

from __future__ import annotations

import unittest

from perfetto_hetero_profiler.schema import SchemaValidationError, record_from_dict
from perfetto_hetero_profiler.schema import jsonschema_runtime

from tests.test_schema_parity_corpus import parity_cases, valid_records


class JsonSchemaRuntimeTests(unittest.TestCase):
    def test_parity_corpus_has_expected_new_decisions(self):
        cases = parity_cases()
        self.assertGreaterEqual(len(cases), 75)
        for name, accepted, data in cases:
            with self.subTest(name=name):
                if accepted:
                    self.assertIsNotNone(record_from_dict(data))
                else:
                    with self.assertRaises(SchemaValidationError):
                        record_from_dict(data)

    def test_validator_is_cached_per_record_type(self):
        jsonschema_runtime._validator.cache_clear()
        event = valid_records()["event"][0]
        record_from_dict(event)
        record_from_dict(dict(reversed(tuple(event.items()))))
        info = jsonschema_runtime.schema_validator_cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (1, 1, 1))

    def test_multiple_errors_are_deterministic_and_sanitized(self):
        event = valid_records()["event"][0]
        first = {**event, "z_unknown": object(), "a_unknown": object()}
        second = dict(reversed(tuple(first.items())))
        errors = []
        for data in (first, second):
            with self.assertRaises(SchemaValidationError) as caught:
                record_from_dict(data)
            errors.append((caught.exception.field_path, caught.exception.message))
        self.assertEqual(errors[0], errors[1])
        self.assertNotIn("object at", str(errors[0]))
        self.assertNotIn("/home/", str(errors[0]))

    def test_only_local_fragment_references_are_accepted(self):
        self.assertEqual(
            list(jsonschema_runtime._walk_references({
                "$ref": "#/$defs/local",
                "$dynamicRef": "#/$defs/dynamic",
            })),
            ["#/$defs/local", "#/$defs/dynamic"],
        )
        self.assertEqual(
            list(jsonschema_runtime._walk_references({"$ref": "https://example.invalid/schema"})),
            ["https://example.invalid/schema"],
        )

    def test_jsonschema_validation_error_is_not_public(self):
        with self.assertRaises(SchemaValidationError) as caught:
            record_from_dict({"record_type": "event"})
        self.assertEqual(type(caught.exception), SchemaValidationError)
