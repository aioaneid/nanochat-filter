from dataclasses import dataclass, field

from nanochat.ltv.ltv_mode import LtvMode


UNBOUNDED_P = 2_147_483_647


class ChunksAndHistory:
    def __init__(self, p: int, q: int, k: int):
        """
        p is only needed when constructing the initial state. That is done externally,
        e.g. in LtvLookBackQkvComputerModule,
        """
        assert p >= 0
        assert q >= 1
        assert k >= 1
        self.p = p
        self.q = q
        self.k = k

    def deque_size(self):
        return 0 if self.is_unbounded() else (self.k - 1 + self.q - 1) // self.q

    def is_unbounded(self):
        return self.p == UNBOUNDED_P

    def to_dict(self):
        return {"p": self.p, "q": self.q, "k": self.k}

    def to_comma_separated_str(self):
        return f"{self.p},{self.q},{self.k}"

    def __str__(self):
        return self.to_comma_separated_str()

    def __repr__(self):
        return f"ChunksAndHistory(p={self.p}, q={self.q}, k={self.k})"

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


UNBOUNDED_LAYER_SPEC = ChunksAndHistory(p=UNBOUNDED_P, q=1, k=1)


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    ltv_r: int = 0
    ltv_query: LtvMode = LtvMode.NONE
    ltv_key: LtvMode = LtvMode.NONE
    ltv_value: LtvMode = LtvMode.NONE
    layer_specs: list[ChunksAndHistory] = field(default_factory=list)

    def __post_init__(self):
        self.layer_specs = [
            ChunksAndHistory.from_dict(spec) if isinstance(spec, dict) else spec
            for spec in self.layer_specs
        ]
        if not self.ltv_r:
            assert self.ltv_query == LtvMode.NONE
            assert self.ltv_key == LtvMode.NONE
            assert self.ltv_value == LtvMode.NONE
            assert not self.layer_specs

    def max_sequence_len(self):
        return self.sequence_len * 10

    def head_dim(self):
        return self.n_embd // self.n_head

    def without_ltv(self):
        return GPTConfig(
            sequence_len=self.sequence_len,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            n_embd=self.n_embd,
            ltv_r=0,
            ltv_query=LtvMode.NONE,
            ltv_key=LtvMode.NONE,
            ltv_value=LtvMode.NONE,
            layer_specs=[],
        )

    def num_ltv_layers(self):
        return len(self.layer_specs)
