# LOB 데이터 안내 (논문 4초안 기준)

- lob.npy: (6288, 10, 4) — 6288 거래일, 10 호가단계, 4채널(bid_price, bid_qty, ask_price, ask_qty). 바이너리라 직접 열어보기 어렵습니다.
- lob_preview.csv: **눈으로 확인용** 미리보기 CSV입니다.
  - time_idx: 거래일 인덱스 (0=첫 거래일, 6287=마지막 거래일).
  - level: 호가 단계 0~9 (10단계).
  - bid_price, bid_qty, ask_price, ask_qty: 매수호가·매수잔량·매도호가·매도잔량.
  - 포함 구간: 처음 50 시점 + 마지막 5 시점 (각 시점당 10레벨 → 총 550행).
- shape: lob_shape.txt 참고. 출처: source.txt (real_file).
