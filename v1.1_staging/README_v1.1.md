# ArtifactBench v1.1 — purged test partition (2026-07-02 audit)

- 원본 test 파티션 2,263 → **2,224** (leak purge 34 + unrecoverable 5)
- 퍼지 근거: bench_origin=test ∩ v9.4 학습 manifest(phase3_balanced_260415) stem 일치 34곡 (전부 YouTube 계열 real)
- 유실 5곡: sonics_real 유튜브 원본 삭제 — track_id는 manifest metadata.missing_unrecoverable에 보존
- 감사 재현: testing/build_verdict28... 아님 — ArtifactNet repo `outputs/artifactbench_test_clean_260702.json` metadata 참조
- 업로드 전 TODO: audit 스크립트 정리 버전 포함, HF dataset v1.1 태그
