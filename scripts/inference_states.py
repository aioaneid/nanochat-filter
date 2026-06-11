import collections

from dataclasses import dataclass


# Equivalent to ltv_look_back_fused_concat_head_major_register_cast_cub_forward_device.
def forward(x0: str, tokens: list[str], p: int, q: int, k: int) -> list[str]:
    result = [None] * len(tokens)
    for i, token in enumerate(tokens):
        if i < p:
            result[i] = (result[i - 1] if i else x0) + token
        elif (i - p) % q == 0:
            window_start = max(0, i - k + 1)
            result[i] = x0 + "".join(tokens[window_start : i + 1])
        else:
            result[i] = result[i - 1] + token
    return result


def compute_p(n: int, p: int, q: int):
    return q - (n - min(p, n)) % q


@dataclass
class State:
    countdown: int
    v: str

    def __post_init__(self):
        assert self.countdown > 0


MAX_SEQ_LEN = 1_000_000


class Cache:
    def __init__(self, x0: str, p: int, q: int, k: int):
        self.x0 = x0
        self.special = State(countdown=p if p else q, v=x0)
        # Ensure minimum deque length of 1 for edge cases where k=1
        # ml = max(1, (k - 1 + q - 1) // q)
        self.q = q
        self.k = k
        if k >= MAX_SEQ_LEN:
            # special state never hits a boundary, ml = 0
            self.special = State(countdown=MAX_SEQ_LEN, v=x0)
            self.d = collections.deque(maxlen=0)  # empty deque
        else:
            # Normal chunked inference
            self.special = State(countdown=p if p else q, v=x0)
            ml = (k - 1 + q - 1) // q
            self.d = collections.deque([x0] * ml, maxlen=ml)

    def advance(self, tokens: list[str]):
        n = len(tokens)
        start = 0

        result = [""] * n

        while start < n:
            al = min(self.special.countdown, n - start)
            # p=skip ensures that there is no boundary
            a = forward(
                self.special.v,
                tokens[start : start + al],
                p=MAX_SEQ_LEN,
                q=self.q,
                k=self.k,
            )
            result[start : start + al] = a

            for i, x in enumerate(self.d):
                # Offset adjusts for the fact that d[i] resolves i boundaries in the future
                offset_i = self.special.countdown + i * self.q - self.k + 1
                bl = max(offset_i, 0)
                kt = tokens[start + bl : start + al]

                # Only run the forward accumulation if there are tokens overlapping the window
                if kt:
                    self.d[i] = forward(x, kt, p=MAX_SEQ_LEN, q=self.q, k=self.k)[-1]

            if al == self.special.countdown:
                new_v = self.d[0] if self.d else self.x0
                self.d.append(self.x0)
                self.special = State(countdown=self.q, v=new_v)
            else:
                self.special = State(countdown=self.special.countdown - al, v=a[-1])

            start += al

        return result
