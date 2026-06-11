import random

def pytest_addoption(parser):
    parser.addoption(
        "--benchmark",
        action="store",
        default="false",
        help="Run in benchmark mode (large sizes)",
    )
    parser.addoption(
        "--sigmoid",
        action="store",
        default="true",
        help="Use sigmoid (expit) activation",
    )
    parser.addoption(
        "--check_against_expected",
        action="store",
        default="true",
        help="Verify correctness against the reference implementation",
    )
    parser.addoption(
        "--shuffle-seed",
        type=int,
        default=11,
        help="Seed for shuffling test order (default: 11)",
    )


def pytest_collection_modifyitems(config, items):
    # Print to confirm the hook is running (remove once verified)
    print(f"\n[pytest_collection_modifyitems] Shuffling {len(items)} items...")
    seed = config.getoption("--shuffle-seed", default=11)
    rng = random.Random(seed)
    rng.shuffle(items)
    print("Shuffle complete.")
