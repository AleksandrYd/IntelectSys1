import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import e2e_s, tpot_s, ttft_s


def main() -> None:
    t0 = 0.0
    t1 = 0.4
    t_last = 2.4
    n_tokens = 101

    ttft = ttft_s(t0, t1)
    tpot = tpot_s(t1, t_last, n_tokens)
    e2e = e2e_s(t0, t_last)

    if tpot is None:
        raise SystemExit("TPOT не должен быть пустым при 101 токене")
    if abs(ttft - 0.4) > 1e-12:
        raise SystemExit(f"TTFT: ожидалось 0.4 с, получено {ttft}")
    if abs(tpot - 0.02) > 1e-12:
        raise SystemExit(f"TPOT: ожидалось 0.02 с, получено {tpot}")
    if abs(e2e - 2.4) > 1e-12:
        raise SystemExit(f"Полное время: ожидалось 2.4 с, получено {e2e}")
    if tpot_s(t1, t_last, 1) is not None:
        raise SystemExit("При одном токене TPOT должен быть None")

    print(f"TTFT = {ttft * 1000:.0f} мс")
    print(f"TPOT = {tpot * 1000:.0f} мс")
    print(f"полное время = {e2e:.1f} с")
    print("проверка арифметики пройдена")


if __name__ == "__main__":
    main()
