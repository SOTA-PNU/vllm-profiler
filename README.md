# Perfetto Heterogeneous Profiler

GPU, RBLN NPU, GPU-prefill/NPU-decode 실행에서 수집한 프로파일링 데이터를
공통 형식으로 정규화하고 Perfetto trace와 HTML 리포트로 변환하는 도구입니다.

## 주요 기능

- GPU/NPU와 host resource telemetry 수집
- 요청, prefill, KV transfer, decode marker 정규화
- E2E, TTFT, TPOT, throughput 및 resource metric 계산
- PyTorch, Nsight Systems, NPU vLLM 상세 event 변환
- RBLN Perfetto trace 검증 및 별도 출력
- deterministic Perfetto trace와 독립 HTML Overview 생성
- schema와 artifact 무결성 검증

누락된 값은 `0`으로 대체하지 않으며, 원본 timestamp와 artifact를 변경하지
않습니다. 값의 availability와 clock alignment 상태도 결과에 함께 기록합니다.

## 설치

Python 3.10 이상이 필요합니다.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
hetero-profiler --help
```

## 빠른 시작

Schema example을 검증합니다.

```bash
hetero-profiler schema validate examples/schema_v1/manifest_hybrid.json
```

Collector 실행 계획을 확인합니다. `--dry-run`은 workload를 시작하지
않습니다.

```bash
hetero-profiler collect gpu \
  --run-root ./runs \
  --run-id example-gpu \
  --profile-mode monitor \
  --dry-run \
  --command python3 -c "print('hello')"
```

GPU Prefill–NPU Decode 통합 실행은 로컬 모델, RBLN cache와 두 vLLM 환경을
미리 준비한 뒤 다음처럼 계획을 확인합니다. 설정 형식은
[`examples/hybrid_config.json`](examples/hybrid_config.json)을 참고하세요.

```bash
hetero-profiler collect hybrid \
  --config /absolute/path/hybrid-config.json \
  --run-root /absolute/path/runs \
  --run-id example-hybrid \
  --profile-mode monitor \
  --dry-run
```

성공한 normalized hybrid run을 Perfetto trace로 변환합니다.

```bash
mkdir -p ./outputs

hetero-profiler convert perfetto \
  --run ./runs/example-hybrid \
  --output ./outputs/example-perfetto \
  --trace-processor /path/to/trace_processor_shell
```

같은 run의 독립 HTML Overview를 생성합니다.

```bash
hetero-profiler overview generate \
  --run ./runs/example-hybrid \
  --perfetto ./outputs/example-perfetto \
  --output ./outputs/example-overview \
  --trace-processor /path/to/trace_processor_shell
```

고정된 Hybrid 구성에서 프로파일러 자체의 정확도·반복성·부하를 검증하는 기능은
설치 패키지가 아니라 저장소 전용 평가 도구로 분리되어 있습니다. 이 도구는 6개
조건의 pilot과 5개 formal round를 사전에 고정하고, 중단 시 checkpoint에서
재개합니다.

```bash
PYTHONPATH=src:. python3 -m tools.evaluation run \
  --config /absolute/path/profiler-experiment-config.json \
  --experiment-root /absolute/path/profiler-experiment \
  --dry-run
```

설정과 실행 정책은 [사용 가이드](docs/usage.md#프로파일러-반복성-및-측정-부하-검증)를
참고하세요. 이 검증은 단일 모델·고정 partition에 대한 제한된 표본이며 일반
하드웨어 benchmark가 아닙니다.

자세한 collector와 변환 명령은 [사용 가이드](docs/usage.md)를 참고하세요.

## 결과

| 결과 | 설명 |
| --- | --- |
| `trace.pftrace` | 처리 단계와 전체 진단 정보를 함께 담는 Perfetto timeline |
| `trace.request-focused.pftrace` | 처리 단계·token boundary·요청 구간 resource 표본·in-window native event를 담는 발표용 timeline |
| `trace.rbln-native.pftrace` | canonical clock에 정렬되지 않은 RBLN native timeline |
| `overview.json` | machine-readable KPI와 provenance |
| `overview.html` | 브라우저에서 여는 self-contained 결과 리포트 |

`trace.pftrace`의 Perfetto **Overview → Info and Stats (advanced)**에는
`vllm_profiler.*` Trace Attribute로 필수 latency, throughput, GPU–NPU
transfer와 resource 요약이 표시됩니다. 측정할 수 없는 단계별 값은 숫자 대신
`not_available`과 reason으로 표시됩니다. 상세 계산
근거, warning과 artifact 목록은 독립 `overview.html`에 유지됩니다.

Perfetto timeline의 `Heterogeneous LLM Processing` 그룹은 실제 marker pair로
관찰한 GPU Prefill, KV 처리·전송, NPU Decode, decode step과 sampling만
표시합니다. E2E/TTFT/TPOT 같은 KPI, Data Quality와 clock 요약을 긴 slice나
counter로 복제하지 않습니다. marker 사이의 근거 없는 간격에도 임의의 work/wait
이름을 붙이지 않습니다. 전체 trace에는 Request lifecycle, 전체 capture resource
telemetry, capture envelope와 native detail을 진단 정보로 유지합니다.
Request-focused trace는 canonical client request 경계에 근거해 request 전
baseline, request와 겹치는 background, request 후 final resource 표본만 선택하며
원본 timestamp나 값을 바꾸지 않습니다.

`overview.html`은 Perfetto UI의 내장 Overview나 plugin이 아닙니다.
`trace.pftrace`는 [Perfetto UI](https://ui.perfetto.dev/)에서 열고,
`overview.html`은 일반 브라우저에서 엽니다.

## vllm-rbln

RBLN NPU에서 vLLM 기반 Hybrid workload를 수집하려면 별도
[`vllm-rbln`](https://github.com/SOTA-PNU/vllm-rbln) 환경이 필요합니다.
기존 normalized run의 검증, 분석, Perfetto 변환과 HTML 생성은 이 저장소만으로
사용할 수 있습니다.

## 지원 범위와 제한사항

- `collect hybrid`는 GPU Prefill, NPU Decode, package proxy, telemetry,
  normalization, Perfetto와 외부 HTML Overview 생성을 한 실행으로 관리합니다.
- `merge hybrid`는 별도의 normalized source 병합 검증 명령이며 실제 workload
  runner가 아닙니다.
- 상세 profiler는 한 run에 하나씩 수집합니다.
- Resource의 단계별 aggregate는 검증된 marker window에서만 계산하며, 발표용
  timeline의 counter 표본과 별개의 요약 값입니다.
- Native profiler clock은 근거가 허용하는 범위에서만 정렬합니다. RBLN
  native trace는 canonical anchor가 없으면 별도 timeline으로 유지합니다.
- HTML Overview는 독립 리포트이며 Perfetto 내장 UI 확장은 제공하지 않습니다.
  UI plugin 통합은 선택적 후속 기능입니다.
- CPU power는 현재 수집하거나 다른 telemetry에서 합성하지 않으며, 측정 지원은
  선택적 후속 기능입니다.
- 단일 smoke run은 통계적 benchmark나 hardware 우열의 근거가 아닙니다.
- 일부 NIXL/UCX 환경에서는 요청과 artifact 생성 후 종료 중 native crash가 발생할
  수 있습니다. 이 경우 결과는 `shutdown_integrity=invalid`, `demo_only=true`이며
  production 성공으로 간주하지 않습니다.

## 문서

- [사용 가이드](docs/usage.md)
- [시스템 설계](docs/design.md)
- [Metric catalog v1](docs/metric_catalog_v1.md)

## 개발

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python3 -m unittest discover -s tests -v

git diff --check
```

## 라이선스

이 저장소는 현재 별도의 오픈소스 라이선스를 제공하지 않습니다.
