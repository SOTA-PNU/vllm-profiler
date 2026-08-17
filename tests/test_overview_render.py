"""Static, deterministic, and accessible Overview HTML tests."""

from __future__ import annotations

import copy
import unittest

from perfetto_hetero_profiler.overview.comparison import build_comparison
from perfetto_hetero_profiler.overview.render import (
    OverviewRenderError,
    render_comparison_html,
    render_overview_html,
    validate_offline_html,
)
from tests.test_overview_comparison import kpi, report


_TEST_CSP = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "script-src 'none'; "
    "connect-src 'none'; "
    "img-src 'none'; "
    "font-src 'none'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)


def resource_summary(
    run_id: str, *, device_type: str, device_id: str, value: float
) -> dict[str, object]:
    aggregate = kpi(
        f"resource.{device_type}.utilization.mean",
        value,
        unit="percent",
        layer="normalized_resource_metric",
    )
    aggregate["scope"].update(
        {
            "run_id": run_id,
            "scope_type": "device",
            "host_id": "host",
            "device_type": device_type,
            "device_id": device_id,
            "window": "capture",
        }
    )
    return {
        "metric_name": f"resource.{device_type}.utilization",
        "canonical_unit": "percent",
        "scope": aggregate["scope"],
        "clock": aggregate["clock"],
        "total_sample_count": 4,
        "available_sample_count": 3,
        "unavailable_sample_count": 1,
        "availability_ratio": 0.75,
        "first_timestamp_ns": 100,
        "last_timestamp_ns": 400,
        "coverage_ns": 300,
        "aggregates": [
            aggregate,
            {
                **copy.deepcopy(aggregate),
                "name": f"resource.{device_type}.utilization.time_weighted_mean",
                "availability": "not_available",
                "value": None,
                "unavailable_reason": "interval coverage is incomplete",
                "sample_count": 0,
            },
        ],
        "quality_warnings": ["resource stream contains unavailable samples"],
    }


class OverviewHTMLTests(unittest.TestCase):
    def rich_report(self, mode: str = "hybrid") -> dict[str, object]:
        item = report("run-overview", run_mode=mode)
        item["resources"] = [
            resource_summary(
                "run-overview", device_type="gpu", device_id="gpu-0", value=0
            ),
            resource_summary(
                "run-overview", device_type="npu", device_id="npu-0", value=2.5
            ),
        ]
        item["native_profiles"] = [
            {
                "kind": "npu_rbln",
                "alignment_status": "partial",
                "parser": "official Trace Processor",
                "policy": "separate unaligned Perfetto trace",
            }
        ]
        return item

    def test_overview_html_is_deterministic_offline_utf8_lf(self):
        source = self.rich_report()
        first = render_overview_html(source)
        second = render_overview_html(copy.deepcopy(source))
        self.assertEqual(first, second)
        self.assertEqual(first.encode("utf-8").decode("utf-8"), first)
        self.assertNotIn("\r", first)
        self.assertTrue(first.endswith("\n"))
        validation = validate_offline_html(first)
        self.assertTrue(validation["valid"], validation["issues"])
        self.assertEqual(validation["network_reference_count"], 0)
        self.assertEqual(validation["absolute_path_count"], 0)

    def test_required_sections_accessibility_and_responsive_table_are_present(self):
        html = render_overview_html(self.rich_report())
        for heading in (
            "Run and workload information",
            "Status and data quality",
            "Request-facing latency",
            "Hybrid pipeline phase breakdown",
            "Throughput and token count",
            "Transfer KPIs",
            "CPU, GPU, and NPU resources",
            "Perfetto trace information",
            "Native profiler policy",
            "Unavailable values and reasons",
            "Provenance and calculation methods",
            "Interpretation cautions",
        ):
            self.assertIn(f">{heading}<", html)
        self.assertIn("<caption>", html)
        self.assertIn('scope="col"', html)
        self.assertIn("Status:", html)
        self.assertIn("overflow-x: auto", html)
        self.assertIn("@media (max-width: 640px)", html)

    def test_zero_and_unavailable_are_visibly_distinct(self):
        html = render_overview_html(self.rich_report())
        self.assertIn("0.000 percent", html)
        self.assertIn("Unavailable — interval coverage is incomplete", html)
        self.assertIn("Unavailable — no classified wait interval", html)

    def test_formula_provenance_scope_and_cautions_are_visible(self):
        html = render_overview_html(self.rich_report())
        self.assertIn("end_timestamp_ns - start_timestamp_ns", html)
        self.assertIn("normalized_metric", html)
        self.assertIn("request_facing_client", html)
        self.assertIn("No randomized repeated trial was performed.", html)
        self.assertIn("RBLN Perfetto payloads are validated", html)

    def test_perfetto_ui_boundary_is_versioned_by_report_semantics(self):
        legacy = self.rich_report()
        legacy_html = render_overview_html(legacy)
        self.assertIn("Independent results dashboard.", legacy_html)
        self.assertIn("not Perfetto's built-in Overview", legacy_html)
        self.assertNotIn("<code>trace.pftrace</code>", legacy_html)

        current = copy.deepcopy(legacy)
        current["interpretation"]["limitations"].append(
            "this external KPI report is not the Perfetto UI; the matching "
            "trace.pftrace contains a separate timeline Heterogeneous LLM "
            "Processing, not the built-in Overview"
        )
        current_html = render_overview_html(current)
        self.assertIn("not the Perfetto UI", current_html)
        self.assertIn("<code>trace.pftrace</code>", current_html)
        self.assertIn("<code>Heterogeneous LLM Processing</code>", current_html)
        self.assertIn("not Perfetto's built-in Overview", current_html)

    def test_all_input_strings_are_escaped_and_sensitive_locations_redacted(self):
        source = self.rich_report()
        source["run"]["run_id"] = "run\"<unsafe>&'"
        source["models"][0]["model_id"] = "/home/person/private/model"
        source["interpretation"]["limitations"].append(
            "See https://example.invalid/report and file:/tmp/private"
        )
        html = render_overview_html(source)
        self.assertIn("run&quot;&lt;unsafe&gt;&amp;&#x27;", html)
        self.assertNotIn("/home/person", html)
        self.assertNotIn("example.invalid", html)
        self.assertNotIn("/tmp/private", html)
        self.assertIn("[redacted absolute path]", html)
        self.assertIn("[redacted URL]", html)
        self.assertTrue(validate_offline_html(html)["valid"])

    def test_request_digests_are_retained_in_json_but_hidden_from_html(self):
        source = self.rich_report()
        prompt_digest = source["workload"]["prompt_sha256"]
        request_digest = source["workload"]["request_body_sha256"]

        html = render_overview_html(source)

        self.assertEqual(source["workload"]["prompt_sha256"], prompt_digest)
        self.assertEqual(source["workload"]["request_body_sha256"], request_digest)
        self.assertNotIn(prompt_digest, html)
        self.assertNotIn(request_digest, html)
        self.assertEqual(
            html.count("Recorded (full SHA-256 retained in overview.json)"),
            2,
        )
        self.assertIn("<td>prompt_sha256</td>", html)
        self.assertIn("<td>request_body_sha256</td>", html)

    def test_nonfinite_available_value_is_rejected(self):
        source = self.rich_report()
        source["kpis"]["request_facing_latency"][0]["value"] = float("nan")
        with self.assertRaisesRegex(OverviewRenderError, "finite"):
            render_overview_html(source)

    def test_gpu_npu_and_hybrid_fixture_modes_have_stable_visible_inventory(self):
        fixtures = {
            "gpu_only": ("gpu", "NVIDIA"),
            "npu_only": ("npu", "Rebellions"),
            "hybrid": ("gpu", "RBLN-CA22"),
        }
        for mode, expected in fixtures.items():
            with self.subTest(mode=mode):
                source = self.rich_report(mode)
                if mode == "gpu_only":
                    source["hardware"] = [
                        item
                        for item in source["hardware"]
                        if item["device_type"] == "gpu"
                    ]
                    source["resources"] = source["resources"][:1]
                elif mode == "npu_only":
                    source["hardware"] = [
                        item
                        for item in source["hardware"]
                        if item["device_type"] == "npu"
                    ]
                    source["resources"] = source["resources"][1:]
                html = render_overview_html(source)
                self.assertIn(f"<td>{mode}</td>", html)
                self.assertIn(expected[0], html)
                self.assertIn(expected[1], html)
                self.assertTrue(validate_offline_html(html)["valid"])


class ComparisonHTMLTests(unittest.TestCase):
    def comparison(self) -> dict[str, object]:
        return build_comparison(
            [
                report("control", request_count=1),
                report(
                    "npu-profile",
                    request_count=1,
                    profile_mode="detailed_profile",
                    profiler_kind="npu_rbln",
                    request_e2e=120,
                ),
            ]
        )

    def test_comparison_html_is_deterministic_and_offline(self):
        source = self.comparison()
        first = render_comparison_html(source)
        second = render_comparison_html(copy.deepcopy(source))
        self.assertEqual(first, second)
        self.assertIn("Independent results dashboard.", first)
        self.assertIn("not Perfetto's built-in Overview", first)
        self.assertTrue(validate_offline_html(first)["valid"])
        self.assertIn("Status: diagnostic_only", first)
        self.assertIn("Baseline: control", first)
        self.assertIn("Absolute delta", first)
        self.assertIn("Percentage delta", first)

    def test_direction_is_explained_without_conclusion_language(self):
        html = render_comparison_html(self.comparison()).casefold()
        self.assertIn("lower_is_preferred", html)
        self.assertIn("higher_is_preferred", html)
        for prohibited in ("winner", "fastest", "best"):
            self.assertNotIn(prohibited, html)

    def test_comparison_strings_are_escaped(self):
        source = self.comparison()
        source["limitations"].append('unsafe "<tag>" & path /home/private')
        html = render_comparison_html(source)
        self.assertIn("&quot;&lt;tag&gt;&quot; &amp;", html)
        self.assertNotIn("/home/private", html)
        self.assertTrue(validate_offline_html(html)["valid"])


class OfflineScannerTests(unittest.TestCase):
    def wrap(self, body: str = "", css: str = "") -> str:
        return (
            "<!doctype html><html><head>"
            '<meta http-equiv="Content-Security-Policy" '
            f'content="{_TEST_CSP}">'
            f"<style>{css}</style></head><body>{body}</body></html>"
        )

    def test_generated_policy_requires_exact_csp(self):
        missing = validate_offline_html("<!doctype html><p>plain</p>")
        self.assertFalse(missing["valid"])
        self.assertIn(
            "exactly one Content-Security-Policy meta element is required",
            missing["issues"],
        )
        permissive = self.wrap().replace(
            "connect-src 'none'", "connect-src *"
        )
        result = validate_offline_html(permissive)
        self.assertFalse(result["valid"])
        self.assertIn("CSP connect-src must be exactly 'none'", result["issues"])

    def test_active_and_container_tags_are_rejected(self):
        tags = ("script", "link", "iframe", "object", "embed", "form")
        for tag in tags:
            with self.subTest(tag=tag):
                result = validate_offline_html(self.wrap(f"<{tag}></{tag}>"))
                self.assertFalse(result["valid"])
                self.assertEqual(result["forbidden_tag_count"], 1)

    def test_url_attributes_and_event_handlers_are_rejected(self):
        for body in (
            '<a href="relative">link</a>',
            '<img src="asset">',
            '<div onclick="work()">event</div>',
        ):
            with self.subTest(body=body):
                result = validate_offline_html(self.wrap(body))
                self.assertFalse(result["valid"])
                self.assertTrue(
                    result["url_attribute_count"]
                    or result["event_handler_count"]
                )

    def test_css_network_constructs_comments_and_escapes_are_rejected(self):
        for css in (
            "body { background: url(asset); }",
            "@import 'theme';",
            "body { background: file:/tmp/item; }",
            "body { color: red; } /* hidden */",
            "body { c\\olor: red; }",
        ):
            with self.subTest(css=css):
                result = validate_offline_html(self.wrap(css=css))
                self.assertFalse(result["valid"])

    def test_text_network_scheme_and_absolute_paths_are_rejected(self):
        for body in (
            "<p>https://example.invalid/item</p>",
            "<p>file:/tmp/item</p>",
            "<p>/home/person/private</p>",
            r"<p>C:\private\item</p>",
        ):
            with self.subTest(body=body):
                result = validate_offline_html(self.wrap(body))
                self.assertFalse(result["valid"])
                self.assertTrue(
                    result["network_reference_count"]
                    or result["absolute_path_count"]
                )


if __name__ == "__main__":
    unittest.main()
