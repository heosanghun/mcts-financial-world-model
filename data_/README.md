# GitHub 업로드용 데이터 (upload_data)

이 폴더는 **GitHub에 드래그 앤 드롭**할 때 사용하는 복사본입니다.  
원본은 `data/` 폴더에 그대로 유지됩니다.

## 포함된 파일

| 폴더 | 파일 |
|------|------|
| **ohlc/** | GSPC.csv, GSPC_source.txt, BTC-USD.csv, BTC-USD_source.txt |
| **macro/** | macro_15.csv, source.txt |
| **lob/** | source.txt, lob_shape.txt, lob.npy, **lob_preview.csv**, README_lob.txt |
| **tick/** | source.txt, tick_shape.txt, tick_sample.csv, tick_sample_with_date.csv, README_tick.txt |
| **news_semantic/** | source.txt, fallback_dates_7.txt, news_semantic_preview_2017-08_onwards.csv, archive-product.yaml |
| **news_semantic/cryptoNewsDataset/** | LICENSE, README.md |

## 사용 방법

1. 이 **upload_data** 폴더 전체를 GitHub 저장소에 드래그 앤 드롭(또는 `data` 폴더로 업로드 후 이 내용으로 교체).
2. 저장소에서 실제 데이터 경로가 `data/` 라면, 업로드 후 `upload_data` 내용을 `data/` 로 복사하거나, 프로젝트 설정에서 경로를 `upload_data` 로 지정하면 됩니다.

- **뉴스 시맨틱 미리보기**: `news_semantic_preview_2017-08_onwards.csv`는 **2017년 8월 1일~2024년 12월 30일** 구간만 포함하며, 논문의 암호화폐·틱 실험 구간(2017.08~2024)과 동일합니다. 파일 상단 각주에 해당 내용이 명시되어 있습니다.

- **기간 점검**: `doc/upload_data_논문기간_단계별_점검보고.md` (형식·기간).
- **내용 점검**: `doc/upload_data_논문내용_점검보고.md` (값·출처·일관성). GitHub 업로드 전 참고.

*생성: data 폴더 업로드 대상만 복사한 폴더*

