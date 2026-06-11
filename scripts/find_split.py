# p is the size of the first chunk
# q is the size of each subsequent non-overlapping chunk
# k is the minimum memory size (including the element itself)
# The first chunk processes p elements, starting from init. The logical warp size is a power-of-two divisor
# of p.
# The other chunks process each q + k - 1 elements, starting from the element before that (or init),
# but output only q elements. The logical warps size is a power-of-two divisor of q + k - 1.
# Example solutions:
#
# redundant: 64 m: 16 k: 5 q: 60 p: 64 k+q-1: 64 *
# Perfectly balanced, plenty of parallelism, two full warps per chunk or four half-warps etc.
#
# redundant: 128 m: 8 k: 17 q: 112 p: 128 k+q-1: 128 *
# Perfectly balanced, sufficient parallelism, four full warps per chunk or eight half-warps etc.
#
# redundant: 992 m: 62 k: 17 q: 16 p: 32 k+q-1: 32 *
#
# Other options with up to twice the redundant work:
# redundant: 336 m: 84 k: 5 q: 12 p: 16 k+q-1: 16 *
# redundant: 200 m: 50 k: 5 q: 20 p: 24 k+q-1: 24 *
# redundant: 416 m: 8 k: 53 q: 108 p: 160 k+q-1: 160 *
# redundant: 704 m: 8 k: 89 q: 104 p: 192 k+q-1: 192 *
# redundant: 992 m: 8 k: 125 q: 100 p: 224 k+q-1: 224 *
#
# For the rotary embeddings it is best if P == Q.

def is_power_of_two(v):
    return v & (v - 1) == 0


def f(t, k):
    p = 0
    while p <= t:
        rest = t - p
        for m in range(1, rest + 1):
            q = rest // m
            q_warp_size = q + k - 1
            if (
                m * q == rest
                and q_warp_size % 8 == 0
                and (q_warp_size < 32 or q_warp_size % 32 == 0)
                and p * 3 // 4 < q_warp_size < p * 5 // 4
            ):
                yield p, m, q
        # Warp logical size at least 8
        p += 8


def redundant_work(m, k):
    return m * (k - 1)


def main():
    t = 1024
    solutions = []
    for k in range(1, t + 1):
        for p, m, q in f(t, k):
            print(
                "k:", k, "p:", p, "m:", m, "q:", q, "redundant:", redundant_work(m, k)
            )
            solutions.append(
                (
                    k,
                    p,
                    m,
                    q,
                )
            )
    sorted_solutions = sorted(
        solutions, key=lambda a: (redundant_work(a[2], a[0]), a[2]), reverse=True
    )
    for a in sorted_solutions:
        k, p, m, q = a
        if m >= 7 and k >= 4:
            print(
                "redundant:",
                redundant_work(m, k),
                "m:",
                m,
                "k:",
                k,
                "q:",
                q,
                "p:",
                p,
                "k+q-1:",
                k + q - 1,
                "*" if p == k + q - 1 else " ",
            )


if __name__ == "__main__":
    main()
