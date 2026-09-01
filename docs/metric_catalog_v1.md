# Metric catalog v1

공식 metric 이름, 단위와 scope를 정리한 문서입니다. Python
`METRIC_CATALOG`가 실제 검증 기준이며, 각 run의 값 제공 여부는
`availability`와 provenance로 확인합니다.

## Metric 목록

| 이름 | 단위 | Scope | 설명 |
| --- | --- | --- | --- |
| `latency.e2e` | ns | request | 요청 시작부터 응답 완료까지의 시간 |
| `latency.prefill` | ns | request, phase | Prefill 구간 |
| `latency.decode` | ns | request, phase | Decode loop 구간 |
| `latency.kv_export` | ns | request, phase | KV export 구간 |
| `latency.kv_transform` | ns | request, phase | KV transform 구간 |
| `latency.kv_transfer` | ns | request, phase, transfer | KV transfer 구간 |
| `latency.sampling` | ns | request, phase | Sampling 구간 합계 |
| `latency.wait` | ns | request, phase | 분류된 wait 구간 합계 |
| `latency.ttft` | ns | request | 첫 token까지의 시간 |
| `latency.tpot` | ns | request | Output token 사이 평균 시간 |
| `throughput.requests` | requests/s | run | 완료 request 처리율 |
| `throughput.input_tokens` | tokens/s | run | Input token 처리율 |
| `throughput.output_tokens` | tokens/s | run | Output token 처리율 |
| `throughput.total_tokens` | tokens/s | run | 전체 token 처리율 |
| `request.input_tokens` | tokens | run, request | Input token 수 |
| `request.output_tokens` | tokens | run, request | Output token 수 |
| `request.total_tokens` | tokens | run, request | 전체 token 수 |
| `request.count` | requests | run | Request 수 |
| `resource.cpu.utilization` | percent | host, process, device | CPU 사용률 |
| `resource.cpu.memory_used` | bytes | host, process, device | CPU/host memory 사용량 |
| `resource.system.memory_used` | bytes | host | System memory 사용량 |
| `resource.gpu.utilization` | percent | device | GPU 사용률 |
| `resource.gpu.memory_used` | bytes | device | GPU memory 사용량 |
| `resource.gpu.power` | W | device | GPU power |
| `resource.npu.utilization` | percent | device | NPU 사용률 |
| `resource.npu.memory_used` | bytes | device | NPU memory 사용량 |
| `resource.npu.power` | W | device | NPU power |
| `transfer.bytes` | bytes | request, transfer | 전송된 byte 수 |
| `transfer.duration` | ns | request, transfer | Transfer 구간 |
| `transfer.handoff_duration` | ns | request | KV export 완료 후 NPU transfer 준비 시작까지의 handoff |
| `transfer.effective_bandwidth` | bytes/s | request, transfer | 실효 전송 bandwidth |
| `transfer.setup_duration` | ns | transfer | NIXL handle 준비 시작부터 비동기 제출 직전까지 |
| `transfer.transform_duration` | ns | request, transfer | Transfer 전 변환 구간 |
| `transfer.wait_duration` | ns | request, transfer | 첫 incomplete 관찰부터 같은 handle의 완료 관찰까지 |
| `transfer.e2e_share` | ratio | request, transfer | E2E 중 transfer 비율 |
| `decode.schedule_wait_duration` | ns | request | KV 사용 가능 후 첫 decode 실행 직전까지 |
| `hybrid.joined_requests` | requests | run | 명시적 ID로 연결된 request 수 |
| `hybrid.unjoined_requests` | requests | run | 연결되지 않거나 ambiguous한 request 수 |
| `hybrid.alignment_offset` | ns | run, host | Source clock offset 추정값 |
| `hybrid.alignment_uncertainty` | ns | run, host | Clock alignment 불확실성 |

## 계산 규칙

- E2E: `response_done - request_received`
- TTFT: `first_token_emitted - request_received`
- TPOT: output token이 두 개 이상일 때
  `(last_token - first_token) / (output_token_count - 1)`
- Effective bandwidth: `transfer.bytes / transfer.duration`
- Transfer E2E share: `transfer.duration / latency.e2e`
- Throughput: 명시적인 measured window의 count를 elapsed time으로 나눈 값

Handoff, setup, transfer와 wait은 서로 다른 runtime 경계를 나타내며 일부 구간은
포함되거나 겹칠 수 있습니다. 따라서 이 값을 더해 전체 transfer latency로
해석하지 않습니다. 첫 status poll에서 이미 완료가 명시적으로 관찰된 경우에만
wait을 `0 ns`로 기록합니다. Marker capability가 없거나 경계가 불완전한 경우에는
0으로 대체하지 않고 `not_available`로 기록합니다.

Start/end pair, token 수, duration 또는 clock alignment 근거가 부족하면 값을
계산하지 않습니다. 나눗셈의 분모가 `0`이거나 unavailable이면 파생 metric도
unavailable입니다.

Resource sample은 metric, host, scope, device와 dimensions가 같은 stream끼리
집계합니다. 서로 다른 GPU/NPU 또는 device의 값을 합산하지 않습니다.

CPU, system memory와 process RSS는 psutil에서 측정합니다. System memory used는
플랫폼별 `used` 값이 아니라 `virtual_memory().total - available`로 계산합니다.
CPU total에서는 Linux user/nice에 이미 포함된 guest/guest_nice를 다시 더하지 않고,
기존 계약과 같이 idle과 iowait를 idle 시간으로 처리합니다.

GPU resource metric은 NVIDIA NVML API에서 측정합니다. Memory는 NVML이 반환한
byte 단위를 그대로 보존하고 power는 milliwatt에서 watt로 변환합니다. GPU 세대에
따라 power가 최근 구간 평균일 수 있으므로 항상 순간값을 뜻하지는 않습니다. 새
sample의 provenance는 `nvml.*` attribute로 기록합니다.

## Availability

- 실제 측정값 `0`은 `available` 값입니다.
- 측정 또는 계산 근거가 없으면 `not_available`입니다.
- 수집하지 않은 항목은 `not_collected`입니다.
- 수집 또는 해석 실패는 `error`입니다.
- 서로 다른 clock domain의 duration은 alignment가 성공한 경우에만
  계산합니다.

## Vendor 확장

공식 catalog 밖의 metric은 `vendor.*`, `vllm.*`, `torch.*`, `nsys.*`,
`rbln.*`, `nixl.*` namespace를 사용합니다. 확장 metric도 unit, scope와
availability를 포함해야 합니다.
