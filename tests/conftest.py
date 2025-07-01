import pytest, os, datetime, sys

# ensure repo root is first on sys.path
TEST_PATH = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(TEST_PATH, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

class Tee:
    """Write everything to multiple file-like objects."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

@pytest.fixture(scope="session", autouse=True)
def redirect_all_output_to_log():
    """
    For every test session, capture stdout/stderr into
    log/pytest-<timestamp>.log *and* still print it to the console.
    """
    # 1. make log dir
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 2. open timestamped log file
    ts = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
    path = os.path.join(log_dir, f"pytest-{ts}.log")
    log_file = open(path, "w")

    # 3. wrap stdout & stderr in a “tee” so prints still go to the console
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(real_stdout, log_file)
    sys.stderr = Tee(real_stderr, log_file)

    # yield control back to pytest — from now on all prints go to both
    yield

    # 4. teardown: restore original streams & close the log
    sys.stdout, sys.stderr = real_stdout, real_stderr
    log_file.close()

# def pytest_cmdline_preparse(config, args):
def pytest_load_initial_conftests(args):
    """
    If pytest is invoked with --collect-only (as VS Code does for discovery),
    strip out any -n / --dist flags so xdist isn't loaded and won't error.
    """
    if "--collect-only" in args:
        filtered = []
        skip_next = False
        for a in args:
            if skip_next:
                skip_next = False
                continue
            if a in ("-n", "--numprocesses"):
                skip_next = True
                continue
            if a.startswith("-n"):
                continue
            filtered.append(a)
        args[:] = filtered
