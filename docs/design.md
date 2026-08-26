# 시스템 설계

Perfetto Heterogeneous Profiler는 GPU/NPU 수집 결과를 공통 schema로
정규화하고, 검증된 Perfetto trace와 HTML 리포트로 변환합니다.

## 데이터 흐름

```text
workload
  ├─ runtime markers
  ├─ resource telemetry
  └─ native profiler artifacts
          │
          ▼
normalized run (schema v1)
          │
          ├─ request correlation
          └─ clock alignment
          │
          ▼
validated hybrid run
          ├─ Perfetto trace
          └─ External KPI Overview
```

주요 구성 요소의 책임은 다음과 같습니다.

- **Collector**: child process lifecycle, telemetry와 raw artifact 수집
- **Schema**: record 구조, 값 범위와 availability 검증
- **Hybrid**: GPU/NPU source의 clock alignment와 request join
- **Perfetto**: event와 metric을 track, slice, counter와 flow로 변환
- **Overview**: KPI와 provenance를 JSON 및 offline HTML로 표현

반복 schedule, checkpoint/resume, 비교 통계와 평가 report는 설치되는 core
package에 포함하지 않습니다. 저장소의 `tools/evaluation`이 공개 core collection
API를 호출하는 얇은 평가 계층으로 이를 제공하며, 평가 compatibility도 그 경계
안에 유지합니다.

입력 run과 raw artifact는 읽기 전용으로 취급합니다. 파생 결과는 새로운
output directory에 생성하며 기존 결과를 덮어쓰지 않습니다.

## Normalized run

Schema v1의 핵심 record는 다음과 같습니다.

| Record | 설명 |
| --- | --- |
| `RunManifest` | run mode, status, workload와 환경 |
| `EventRecord` | request 또는 phase의 instant/span |
| `MetricSample` | availability를 포함한 metric sample |
| `ArtifactReference` | run-relative artifact identity |
| `ClockDomain` | timestamp의 clock domain |
| `SyncPoint` | 두 clock 사이의 관측점 |
| `ClockTransform` | source timestamp의 변환 정보 |

시간은 정수 nanosecond로 저장합니다. 알 수 없는 top-level field, 음수
timestamp/duration, NaN과 Infinity는 거부합니다. 도구별 추가 정보는
namespaced attribute로 기록합니다.

수집하지 못한 값은 다음 availability로 표현합니다.

- `available`: 유한한 값이 존재함. 실제 값 `0`도 포함
- `not_available`: 계산 또는 측정 근거가 없음
- `not_collected`: 수집을 수행하지 않음
- `error`: 수집 또는 해석 중 오류 발생

## Runtime markers와 correlation

Hybrid timeline은 request, prefill, KV export/transform/transfer, decode,
sampling과 token emission marker로 구성됩니다. 관측되지 않은 marker나
timestamp를 생성하지 않습니다.

GPU와 NPU request는 다음과 같은 명시적인 식별자로만 연결합니다.

1. request ID
2. transfer ID
3. correlation ID

식별자가 없거나 여러 candidate와 일치하면 임의로 연결하지 않습니다.
Timestamp 또는 event name의 유사성은 correlation 근거로 사용하지 않습니다.

## Clock alignment

서로 다른 clock domain의 timestamp는 변환 근거가 있을 때만 비교합니다.
Clock transform에는 method, offset과 uncertainty를 함께 기록합니다.

```text
canonical_ns = source_ns + offset_ns
```

Alignment uncertainty가 허용 범위를 넘으면 duration을 `0`으로 만들지 않고
unavailable 또는 partial 상태로 유지합니다. 원본 timestamp와 clock domain은
항상 보존합니다.

Native profiler clock에 canonical anchor가 없으면 request boundary에 맞춰
임의로 rebase하지 않습니다. 이런 trace는 별도 native timeline으로
게시합니다.

## Metrics and availability

공식 metric은 Python `METRIC_CATALOG`에서 정의합니다. Latency, throughput,
request count, resource, transfer와 alignment 품질을 포함합니다. 전체 목록과
단위는 [Metric catalog](metric_catalog_v1.md)를 참고하세요.

계산 원칙은 다음과 같습니다.

- Latency는 검증된 start/end event pair로 계산
- TTFT는 request 시작부터 첫 token까지 계산
- TPOT는 output token이 두 개 이상일 때만 계산
- Throughput은 명시적인 measured window 사용
- Resource metric은 host, device와 dimensions별로 분리
- 실제 `0`, missing과 unavailable을 구분
- Request-facing latency와 canonical pipeline latency를 별도 계층으로 유지

## Native profiler 지원

| Profiler | 입력 | 변환 결과 | Alignment |
| --- | --- | --- | --- |
| GPU PyTorch/Kineto | Chrome trace JSON | ATen, CUDA API, kernel, memory event | `partial_derived` |
| GPU Nsight Systems | official SQLite export | NVTX, CUDA API, kernel, memory event | `partial_derived` |
| NPU vLLM | Chrome trace JSON | ATen, compiled region, host event | `partial_derived` |
| NPU RBLN | Perfetto protobuf | 별도 native Perfetto trace | `partial_unaligned` |

Native event는 source format이 제공하는 process, thread, stream, category와
correlation ID를 보존합니다. 확인되지 않은 device execution, parent 관계나
API-to-kernel flow를 생성하지 않습니다.

RBLN aggregate는 표준 Perfetto trace로 검증합니다. Canonical anchor가 없으면
원본 payload를 `trace.rbln-native.pftrace`로 별도 게시하고 hybrid timeline에
합치지 않습니다.

## Perfetto output

Canonical trace는 다음 request 중심 hierarchy를 사용합니다.

```text
Hybrid Request
├─ GPU Prefill
├─ KV Export
├─ KV Handoff
├─ KV Transfer Setup
├─ KV Transfer
├─ KV Transfer Wait
├─ KV Transform
├─ Decode Scheduling Wait
└─ NPU Decode
```

추가 구간은 versioned runtime marker가 있을 때만 생성합니다. KV Handoff는 GPU
KV export 완료부터 NPU가 transfer handle 준비를 시작할 때까지, Transfer Setup은
handle 준비 시작부터 비동기 transfer 제출 직전까지입니다. Transfer Wait은 제출
후 첫 `PROC` 관찰부터 동일 handle의 `DONE` 관찰까지이며 polling 간격만큼 경계
오차가 있을 수 있습니다. 첫 poll이 `DONE`이면 명시적으로 관찰된 0이고, marker가
없는 상태와 구분합니다. Decode Scheduling Wait은 KV 동기화 완료 후 첫 decode
model 실행 직전까지이며 scheduler 전체 queue 체류 시간을 뜻하지는 않습니다.

Setup과 Wait은 전체 Transfer 구간에 포함될 수 있으므로 단순 합산하지 않습니다.
모든 pair는 같은 request, correlation ID, transfer ID와 clock domain을 요구합니다.
Capability가 없는 이전 run은 기존 계약으로 검증하며 새 KPI는
`not_available`로 유지합니다.

Resource metric은 counter track으로, native operation은 profiler와
process/thread/device 단위의 하위 track으로 구성합니다. 선택적인
request-focused trace는 client request 시작/종료를 직접 변환한 canonical window를
사용합니다. 각 resource stream에서 request 전 baseline, sampling interval이
request와 겹치는 background, request 후 final 표본만 선택합니다. Full-capture
telemetry를 복제하거나 timestamp와 값을 다시 기준화하지 않습니다.

Trace Processor validation은 trace의 track, slice, counter, annotation과
flow를 변환 계획과 대조합니다. 동일한 입력과 설정은 동일한 protobuf output을
생성해야 합니다.

## Artifact integrity

파생 결과에는 SHA-256 기반 detached artifact manifest와 validation이
포함됩니다. Artifact path는 output root 기준 상대경로로 기록하며 절대경로,
상위 directory 이동과 symlink를 허용하지 않습니다.

Publication은 다음 원칙을 따릅니다.

- source와 output 경로 분리
- 기존 output 덮어쓰기 금지
- regular file만 허용
- 변환 전후 input identity 확인
- 모든 payload 검증 후 결과 게시

## External KPI Overview

Overview는 normalized run과 matching Perfetto output을 검증한 뒤 만드는 독립
JSON/HTML 리포트입니다. Perfetto UI의 내장 Overview 또는 plugin이 아닙니다.

리포트는 KPI의 availability, 계산식, sample count와 provenance를 보존합니다.
비교 조건이 맞지 않으면 성능 순위나 승자를 생성하지 않습니다.

Perfetto Info and Stats의 Trace Attribute schema `1.1.0`은 중복된 KPI별
`availability` 행을 출력하지 않습니다. 숫자 value는 available을 의미하고,
동일한 value key의 `not_available` 문자열은 측정 근거가 없음을 의미합니다.
정수 `0`은 실제 관측값으로 그대로 유지됩니다. sample count와 aggregation은
Trace Attribute에 유지되며, canonical availability와 상세 provenance는
normalized metric 및 외부 HTML/JSON Overview에 그대로 남습니다.
새 trace의 namespace는 정확히 `vllm_profiler.`입니다. 요청 측정 E2E와
marker 기반 pipeline E2E는 각각 `kpi.latency.e2e`와
`pipeline.latency.e2e` 아래에 분리합니다. Timeline은 실제 marker pair와 검증된
token timestamp만 사용하고, KPI 숫자는 Info and Stats에만 기록합니다.

## 제한사항

- Public `merge hybrid` 명령은 synthetic source 검증용입니다.
- 실제 GPU Prefill–NPU Decode 실행은 `collect hybrid`가 서버, proxy, telemetry,
  normalization과 결과 생성을 함께 관리합니다. 환경별 model/cache/vLLM 설정은
  사용자가 제공해야 합니다.
- 추가 transfer 구간은 versioned runtime marker capability가 있는 run에서만
  제공하며, 이전 run은 추정하지 않습니다.
- 단계별 resource aggregate는 marker window와 sampling coverage가 충분할 때만
  제공하며, 짧은 peak는 polling interval 사이에서 누락될 수 있습니다.
- Native clock alignment는 exact synchronization이 아닐 수 있습니다.
- RBLN native trace는 canonical anchor가 없으면 별도 timeline입니다.
- Perfetto UI 내장 Overview와 custom plugin은 제공하지 않으며, plugin 통합은
  선택적 후속 기능입니다.
- CPU power collector는 제공하지 않고 측정되지 않은 값을 합성하지 않으며,
  측정 지원은 선택적 후속 기능입니다.
- 상세 profiler는 상호 간섭을 피하기 위해 한 run에 하나만 활성화합니다.
- 검증은 고정 model/partition과 제한된 표본을 사용하므로 일반화된 hardware
  benchmark나 우열을 의미하지 않습니다.
- 일부 NIXL/UCX 환경의 종료 crash는 요청 성공과 분리해
  `shutdown_integrity=invalid`, `demo_only=true`로 기록하며 production 성공으로
  판정하지 않습니다.
