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
| `hetero-profiler merge hybrid` | synthetic GPU/NPU source 병합 검증 |
| `hetero-profiler convert perfetto` | normalized hybrid run을 Perfetto로 변환 |
| `hetero-profiler overview generate` | 단일 run의 JSON/HTML 리포트 생성 |
| `hetero-profiler overview compare` | 여러 Overview 비교 |

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

## Hybrid source

현재 `merge hybrid` 명령은 clock alignment와 request join을 테스트하는
synthetic source 전용입니다. 실제 GPU-prefill/NPU-decode workload runner가
아닙니다.

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
conversion_manifest.json
artifact_manifest.json
artifact_manifest_validation.json
```

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
