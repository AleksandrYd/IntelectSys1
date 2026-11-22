def ttft_s(t0: float, t1: float) -> float:
    """Время до первого токена: t1 - t0."""
    return t1 - t0


def tpot_s(t1: float, t_last: float, n_tokens: int) -> float | None:
    """Среднее время одного токена после первого. При n <= 1 вернуть None."""
    if n_tokens <= 1:
        return None
    return (t_last - t1) / (n_tokens - 1)


def e2e_s(t0: float, t_last: float) -> float:
    """Полное время ответа: tN - t0."""
    return t_last - t0
