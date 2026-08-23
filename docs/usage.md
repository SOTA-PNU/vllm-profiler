# 사용 가이드

이 문서는 설치, 수집, 변환 및 결과 확인의 기본 흐름을 설명합니다. 전체 옵션은
각 명령의 `--help`에서 확인하세요.

## 설치

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
hetero-profiler --help
```

저장소에서 직접 실행할 수도 있습니다.

```bash
PYTHONPATH=src python3 -m perfetto_hetero_profiler --help
```

Perfetto 변환에는 package에 고정된 `perfetto`, `protobuf`와 호환되는
`trace_processor_shell`이 필요합니다. 재현 가능한 실행에서는 사용할
binary를 `--trace-processor`로 명시하세요.

## CLI

| 명령 | 설명 |
| --- | --- |
| `hetero-profiler schema` | schema 조회 및 JSON/JSONL 검증 |
| `hetero-profiler collect gpu` | child command와 GPU/host telemetry 수집 |
| `hetero-profiler collect gpu-vllm` | 로컬 vLLM request와 GPU telemetry 수집 |
| `hetero-profiler collect npu` | child command와 RBLN NPU/host telemetry 수집 |
| `hetero-profiler collect npu-runtime` | compiled RBLN artifact의 runtime 수집 |
| `hetero-profiler collect hybrid` | GPU Prefill–NPU Decode 통합 실행과 결과 생성 |
| `hetero-profiler merge hybrid` | synthetic GPU/NPU source 병합 검증 |
| `hetero-profiler convert perfetto` | normalized hybrid run을 Perfetto로 변환 |
| `hetero-profiler overview generate` | 단일 run의 JSON/HTML 리포트 생성 |
| `hetero-profiler overview compare` | 여러 Overview 비교 |
| `hetero-profiler phase7` | 고정 Hybrid 조건의 정확도·반복성·측정 부하 검증 |

실행 전 `--dry-run`으로 경로와 child command를 확인하는 것을 권장합니다.

## Schema 검증

```bash
hetero-profiler schema version
hetero-profiler schema list
hetero-profiler schema validate examples/schema_v1/manifest_hybrid.json
```

## GPU 수집

### 일반 command monitor

`collect gpu`는 child command를 실행하면서 GPU, CPU, system memory와 child
memory를 수집합니다.

```bash
hetero-profiler collect gpu \
  --run-root ./runs \
  --run-id example-gpu \
  --profile-mode monitor \
  --sample-interval-ms 1000 \
  --timeout-sec 60 \
  --dry-run \
  --command python3 -c "print('hello')"
```

`--command` 뒤의 인자는 shell string이 아니라 child argv로 처리됩니다.

### GPU vLLM

`collect gpu-vllm`은 loopback vLLM server를 시작하고 readiness, warm-up,
streaming request, telemetry와 cleanup을 관리합니다. 모델과 vLLM 실행 환경은
미리 준비해야 합니다.

```bash
hetero-profiler collect gpu-vllm \
  --run-root "$(pwd)/runs" \
  --run-id example-gpu-vllm \
  --model /path/to/local-model \
  --vllm-bin /path/to/vllm \
  --profile-mode monitor \
  --host 127.0.0.1 \
  --port 18080 \
  --sample-interval-ms 500 \
  --warmup-requests 1 \
  --measured-requests 2 \
  --max-output-tokens 8 \
  --offline \
  --dry-run
```

지원 mode는 `monitor`, `torch`, `nsys`입니다. PyTorch Profiler와 Nsight
Systems는 별도 run으로 실행해야 합니다.

## NPU 수집

### 일반 command monitor

`collect npu`는 child command와 함께 `rbln-smi --json`, CPU, system memory와
child memory를 수집합니다.

```bash
hetero-profiler collect npu \
  --run-root ./runs \
  --run-id example-npu \
  --profile-mode monitor \
  --device-id 0 \
  --sample-interval-ms 1000 \
  --timeout-sec 60 \
  --dry-run \
  --command python3 -c "print('hello')"
```

### Direct RBLN runtime

`collect npu-runtime`은 이미 compile된 RBLN artifact를 사용합니다. Artifact와
runtime Python의 SDK/compiler/target 호환성은 사용자가 준비해야 합니다.

```bash
hetero-profiler collect npu-runtime \
  --run-root "$(pwd)/runs" \
  --run-id example-npu-runtime \
  --artifact /path/to/model.rbln \
  --runtime-python /path/to/runtime-python \
  --profile-mode monitor \
  --device-id 0 \
  --warmup-inferences 3 \
  --measured-inferences 10 \
  --timeout-sec 60 \
  --dry-run
```

상세 RBLN capture는 고유한 `run-id`와 `--profile-mode detailed-profile`을
사용합니다. Token stream이 없는 workload에서는 TTFT와 TPOT가
`not_available`일 수 있습니다.

## GPU Prefill–NPU Decode 통합 실행

`collect hybrid`는 세 child process를 모두 package 코드로 시작하고 소유합니다.
설정은 절대 경로를 사용하며 출력은 반드시 새 run ID여야 합니다. 먼저 제공된
[`hybrid_config.json`](../examples/hybrid_config.json)을 복사해 로컬 model,
cache, vLLM Python 환경과 Perfetto/Nsight 경로를 수정합니다.

```bash
hetero-profiler collect hybrid \
  --config /absolute/path/hybrid-config.json \
  --run-root /absolute/path/runs \
  --run-id manual-hybrid-01 \
  --profile-mode monitor \
  --dry-run
```

`--dry-run`을 제거하면 readiness, telemetry, warm-up, 측정, normalization,
Perfetto 변환과 외부 HTML Overview 생성을 순서대로 수행합니다. `--prompt`,
`--prompt-file`, `--warmup-requests`, `--measured-requests`,
`--max-output-tokens`으로 workload 일부를 덮어쓸 수 있습니다. Prompt와 생성
텍스트는 결과에 저장하지 않습니다.

Profile mode는 다음 중 하나만 선택할 수 있습니다.

| Mode | 상세 수집 대상과 권장 용도 |
| --- | --- |
| `monitor` | canonical marker와 CPU/GPU/NPU/memory telemetry를 함께 보는 기본 실행 |
| `gpu-torch` | GPU Prefill의 PyTorch/Kineto operator와 framework 활동 분석 |
| `gpu-nsys` | GPU Prefill의 CUDA API, kernel, memcpy와 공식 correlation 분석 |
| `npu-torch` | NPU Decode server의 host-side PyTorch/ATen 활동 분석 |
| `npu-rbln` | RBLN device Neural Engine/DMA를 native Perfetto timeline에서 분석 |

한 실행에서 profiler 하나만 켜는 이유는 profiler끼리 lifecycle과 장치 상태를
간섭시키지 않고 각 capture의 overhead와 provenance를 분리하기 위해서입니다.
NPU Torch는 NPU 내부 실행 시간이 아니라 Decode server의 host-side 활동입니다.

기존 immutable-source 정책 때문에 결과는 sibling 디렉터리로 분리됩니다.

```text
runs/<run-id>/              normalized hybrid bundle
runs/<run-id>-gpu/          immutable GPU source and raw capture
runs/<run-id>-npu/          immutable NPU source and raw capture
runs/<run-id>-coordinator/  server logs, cleanup and validation evidence
runs/<run-id>-perfetto/     trace.pftrace and detached validation
runs/<run-id>-perfetto-request-focused/ presentation trace bundle
runs/<run-id>-overview/     external overview.json and overview.html
runs/<run-id>-closeout-recovery/ detached immutable-input manifest
runs/<run-id>-publication/  overall result and determinism evidence
```

실패 원인은 `<run-id>-coordinator/result.json`과 `raw/*.stderr.log`에서
확인합니다. Runner는 leader의 정상 종료를 먼저 요청하고, 필요할 때만 자신이
만든 process group을 단계적으로 정리합니다. 기존 서버나 다른 사용자 process는
종료하지 않습니다.

RBLN PB는 공식 Perfetto trace이지만 canonical `CLOCK_MONOTONIC` anchor가 없는
경우 `trace.rbln-native.pftrace`로 분리됩니다. Relative timestamp를 임의로
Hybrid timeline에 이동하지 않습니다. HTML Overview는 Perfetto UI plugin이
아닌 독립적인 결과 화면입니다. 한 번의 smoke 실행은 benchmark가 아닙니다.
현재 canonical marker에는 KV transfer 전체 구간은 있지만 transfer setup과
wait를 독립적으로 분리하는 marker는 없습니다.

## Phase 7B 프로파일러 검증

`phase7`은 `collect hybrid`를 반복 호출해 프로파일러 자체의 정확도, 반복성,
측정 부하를 검증합니다. 설정 예시는
[`phase7b_config.json`](../examples/phase7b_config.json)입니다. 먼저 고정
Hybrid 설정을 준비하고 그 파일의 SHA-256을 Phase 7B 설정에 기록합니다.

```bash
sha256sum /absolute/path/hybrid-config.json

hetero-profiler phase7 run \
  --config /absolute/path/phase7b-config.json \
  --experiment-root /absolute/path/phase7b-experiment \
  --dry-run
```

`--dry-run`은 파일, 서버, 포트를 사용하지 않고 36개 logical trial과 최대 42개
hardware attempt 계획만 출력합니다. 실제 실행은 `--dry-run`을 제거합니다.
중단된 실험은 같은 설정과 출력 경로로 재개합니다.

```bash
hetero-profiler phase7 run \
  --config /absolute/path/phase7b-config.json \
  --experiment-root /absolute/path/phase7b-experiment \
  --resume

hetero-profiler phase7 status \
  --experiment-root /absolute/path/phase7b-experiment

hetero-profiler phase7 validate \
  --experiment-root /absolute/path/phase7b-experiment

hetero-profiler phase7 report \
  --experiment-root /absolute/path/phase7b-experiment
```

`phase7 validate`는 성공 trial을 현재 디스크 상태에서 다시 읽어 검증 결과를
표준 출력으로 반환하며 기존 experiment 파일을 수정하지 않습니다.

이미 게시된 experiment report를 보존하면서 report 집계 코드를 다시 적용하려면
겹치지 않는 새 출력 경로를 지정합니다. 이 명령은 source trial을 읽기 전용으로
사용하고 report, limitations, source provenance와 detached manifest만 게시합니다.

```bash
hetero-profiler phase7 report \
  --experiment-root /absolute/path/phase7b-experiment \
  --output-root /absolute/path/phase7b-report
```

실험 조건은 `reference`, `monitor`, `gpu_torch`, `gpu_nsys`, `npu_torch`,
`npu_rbln`입니다. Reference는 resource collector와 상세 profiler를 끄지만 현재
runtime marker emission은 남으므로 완전한 무계측 기준이 아닙니다. 각 조건은
pilot 1회와 formal 5회로 실행되고, pilot은 formal 통계에서 제외됩니다.

독립 streaming client의 `CLOCK_MONOTONIC_NS` 원본 경계로 E2E, TTFT, TPOT를
재계산하며 요청·토큰·marker는 정확히 대조합니다. Formal 결과에는 표본
표준편차(`n-1`), CV, MAD, p50, p95와 같은 round의 paired overhead가
포함됩니다. `report.html`은 Perfetto 내장 Overview가 아닌 독립 결과
dashboard입니다. 5회 formal 반복과 단일 모델·고정 partition 결과를 일반적인
benchmark 또는 하드웨어 우열로 해석하면 안 됩니다.

검증된 고정 실행의 최초 report는 postprocess 집계에서 NPU source telemetry가
누락되어 superseded 처리했습니다. 원본 experiment와 raw artifact는 변경하지
않았고, 같은 36개 성공 trial에 수정된 집계만 다시 적용한 corrected report를
canonical publication으로 사용합니다. 이는 hardware 재실행 결과가 아닙니다.
Monitor와 reference의 E2E 차이 약 `-0.37%`는 formal 5회 안의 측정 변동으로
해석하며 성능 향상으로 표현하지 않습니다. Reference 역시 runtime marker
emission이 남아 있으므로 완전한 무계측 또는 순수 원본 성능 기준이 아닙니다.

## Hybrid source 병합

`merge hybrid` 명령은 clock alignment와 request join을 테스트하는 synthetic
source용 저수준 명령입니다. 실제 실행에는 위의 `collect hybrid`를 사용합니다.

```bash
hetero-profiler merge hybrid \
  --run-root "$(pwd)/runs" \
  --run-id example-hybrid \
  --gpu-run "$(pwd)/runs/gpu-source" \
  --npu-run "$(pwd)/runs/npu-source" \
  --alignment-method fake \
  --dry-run
```

GPU/NPU request는 명시적인 request, transfer 또는 correlation ID로만
연결합니다. Timestamp 근접성만으로 연결하지 않습니다.

## Run 구조

수집 결과는 새 `<run-root>/<run-id>`에 생성됩니다.

```text
runs/<run-id>/
├── manifest.json
├── clocks/
├── events/events.jsonl
├── metrics/metrics.jsonl
├── artifacts/artifacts.jsonl
├── raw/
└── summary/
```

Manifest status는 `succeeded`, `partial`, `failed` 중 하나입니다. 개별 record는
다음처럼 다시 검증할 수 있습니다.

```bash
hetero-profiler schema validate ./runs/example/manifest.json
hetero-profiler schema validate ./runs/example/events/events.jsonl
```

## Perfetto 변환

`convert perfetto`는 검증된 `succeeded` hybrid run을 읽고 새 output에 결과를
생성합니다. Source와 output은 겹칠 수 없으며 기존 output을 덮어쓰지
않습니다.

```bash
mkdir -p ./outputs

hetero-profiler convert perfetto \
  --run ./runs/example-hybrid \
  --output ./outputs/example-perfetto \
  --trace-processor /path/to/trace_processor_shell \
  --dry-run
```

기본 output은 다음과 같습니다.

```text
trace.pftrace
trace_validation.json
trace_attributes_validation.json
conversion_manifest.json
artifact_manifest.json
artifact_manifest_validation.json
```

`trace.pftrace`를 Perfetto UI에서 연 뒤 **Overview → Info and Stats
(advanced)**를 선택하면 `vllm_profiler.*` namespace의 필수
성능 요약을 볼 수 있습니다. Timeline은 시간에 따른 event와 resource counter를,
별도 `overview.html`은 전체 KPI 근거, warning, availability와 artifact 정보를
담습니다. Info and Stats는 HTML Overview를 대체하지 않습니다.

Trace Attribute는 공식 protobuf가 지원하는 string과 signed int64만 사용합니다.
Rate는 원래 값을 1,000배 한 `milli-per-second`, utilization은 1,000배 한
`milli_percent`, E2E share는 ratio를 percent로 바꾼 뒤 1,000배 한
`milli_percent`로 기록합니다. 정수 변환은 decimal half-even 반올림입니다.
Trace Attribute schema `1.1.0`에서는 KPI마다 별도의 `availability` 행을 만들지
않습니다. 숫자 value는 측정 가능한 값이고, 같은 value key의 문자열
`not_available`은 측정 근거가 없다는 뜻입니다. 따라서 실제 정수 `0`과
`not_available`은 서로 다른 상태입니다. 필요한 unavailable reason은 함께
표시합니다. Prefill, Transfer, Decode resource는 canonical 계산 계층에 해당
marker window가 존재할 때만 기록하며, capture 전체 값을 단계 값으로 복사하지
않습니다.
Capture 전체 resource aggregate는 `vllm_profiler.resource.capture.*`로 분리합니다.
요청 E2E는 `vllm_profiler.kpi.latency.e2e.*`, marker 기반 pipeline E2E는
`vllm_profiler.pipeline.latency.e2e.*`로 분리하여 서로 덮어쓰지 않습니다.
종료 로그에서 검증된 fatal signature가 발견된 입력은 `demo_only=true`와
`source.shutdown_integrity=invalid`로 표시하므로 정상 benchmark로 해석하면 안
됩니다.
상세 availability, sample count, aggregation과 provenance는 외부 HTML/JSON
Overview에서 확인할 수 있습니다.

상세 profiler event와 요청 중심 trace가 필요하면 옵션을 추가합니다.

```bash
hetero-profiler convert perfetto \
  --run ./runs/example-hybrid \
  --output ./outputs/example-detailed-perfetto \
  --trace-processor /path/to/trace_processor_shell \
  --include-native-details \
  --request-focused \
  --dry-run
```

RBLN trace에 canonical clock anchor가 없으면
`trace.rbln-native.pftrace`로 별도 생성되며 timestamp를 임의로 이동하지
않습니다.

Versioned transfer marker가 포함된 hybrid run에서는 request group 아래의
`KV Handoff`, `KV Transfer Setup`, `KV Transfer Wait`,
`Decode Scheduling Wait` track을 확인할 수 있습니다. Wait은 status polling으로
관찰한 구간이므로 실제 device completion보다 최대 polling 간격만큼 늦게 끝날 수
있습니다. Setup, transfer, wait은 중첩될 수 있어 합산 값이 아닙니다. 이전 run처럼
marker capability가 없으면 가짜 slice를 만들지 않으며 해당 KPI를
`not_available`로 표시합니다. 첫 poll에서 완료가 확인된 경우에만 wait은 관찰된
`0 ns`입니다.

## HTML Overview

Overview는 Perfetto UI와 별개인 JSON/HTML 결과 리포트입니다.

```bash
hetero-profiler overview generate \
  --run ./runs/example-hybrid \
  --perfetto ./outputs/example-perfetto \
  --output ./outputs/example-overview \
  --trace-processor /path/to/trace_processor_shell \
  --dry-run
```

현재 Overview 입력은 native detail option 없이 만든 기본 Perfetto bundle이어야
합니다. 결과에는 `overview.json`, `overview.html`, validation과 detached
artifact manifest가 포함됩니다.

Transfer 표에는 Handoff duration, Transfer setup duration, Transfer completion
wait, Decode scheduling wait이 값, availability, sample 수와 source marker와
함께 표시됩니다. 값이 `0`이면 명시적인 완료 관찰이며, `not_available`은 marker
capability 또는 유효한 pair가 없다는 뜻입니다.

Resource 표는 Prefill, Transfer, Decode와 capture 전체를 분리합니다. Utilization과
power는 첫 synthetic interval을 제외한 trailing interval
`(timestamp_ns - interval_ns, timestamp_ns]`과 marker window의 overlap으로
time-weighted mean을 계산합니다. 완전한 interval coverage가 없으면 값은
`not_available`입니다. Memory는 point-in-time gauge이므로 stage 안에 실제
timestamp가 있는 sample만 peak에 사용하며 hold-last나 보간을 하지 않습니다.
Sampling interval이 stage보다 길면 계산 가능한 값에도
`value is not stage-exclusive` warning이 붙습니다.

실제 stage 값을 수집하려면 measured request 전 baseline sample, 처리 중 background
sample, response 후 collector 종료 전 final sample이 모두 필요합니다. Sampling
interval은 분석하려는 가장 짧은 stage보다 충분히 짧아야 합니다. 이 조건이 없는
기존 run은 가까운 sample이나 capture 평균으로 채우지 않습니다.

여러 Overview를 비교할 수 있습니다.

```bash
hetero-profiler overview compare \
  --input ./outputs/control-overview \
  --input ./outputs/profile-overview \
  --output ./outputs/overview-comparison \
  --dry-run
```

단일 request나 profiler 종류가 다른 결과의 비교는 진단용이며 benchmark로
해석하지 않습니다.

## 결과 열기

- `trace.pftrace`: [Perfetto UI](https://ui.perfetto.dev/)의 **Open trace file**
- `trace.rbln-native.pftrace`: Perfetto UI에서 별도 native timeline으로 열기
- `overview.html`, `comparison.html`: 일반 브라우저의 **Open File**
- `.pt.trace.json.gz`: 지원되는 Perfetto UI에서 원본 trace 열기
- `.nsys-rep`: NVIDIA Nsight Systems에서 열기
- JSON/JSONL: `jq` 또는 schema validator로 확인

HTML은 self-contained offline 파일입니다. 공개 HTML에는 workload digest의
기록 여부만 표시하고 전체 SHA-256은 `overview.json`에 유지합니다.

## 안전 규칙

- 고유한 `run-id`와 존재하지 않는 output을 사용합니다.
- 실행 전 `--dry-run`으로 명령과 경로를 확인합니다.
- 다른 사용자의 GPU/NPU process를 종료하지 않습니다.
- Raw log를 공유하기 전에 credential, 로컬 경로, prompt/response와 장치
  식별 정보를 확인합니다.
- 누락되거나 오류인 값을 `0`으로 바꾸지 않습니다.
- Native timestamp를 근거 없이 canonical clock으로 이동하지 않습니다.
