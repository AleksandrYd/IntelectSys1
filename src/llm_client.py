import time

from openai import OpenAI

from metrics import e2e_s, tpot_s, ttft_s


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def timed_chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0,
    ignore_eos: bool = False,
) -> dict:
    """Один потоковый запрос. Время измеряется на стороне клиента."""
    extra_body = {"ignore_eos": True} if ignore_eos else None
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body,
    )
    t1 = None
    t_last = None
    usage = None
    parts: list[str] = []
    finish_reason = None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        content = choice.delta.content if choice.delta is not None else None
        if content:
            now = time.perf_counter()
            if t1 is None:
                t1 = now
            t_last = now
            parts.append(content)
    if usage is None or t1 is None or t_last is None:
        raise RuntimeError("сервер не прислал текст или поле usage")
    n = usage.completion_tokens
    return {
        "ttft_s": ttft_s(t0, t1),
        "tpot_s": tpot_s(t1, t_last, n),
        "e2e_s": e2e_s(t0, t_last),
        "prompt_tokens": usage.prompt_tokens,
        "output_tokens": n,
        "finish_reason": finish_reason,
        "text": "".join(parts),
    }