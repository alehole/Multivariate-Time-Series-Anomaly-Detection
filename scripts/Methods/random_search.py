from pathlib import Path
import subprocess
import sys
import time

SCRIPTS = [
    Path("Blackbox/RNN/train_rnn_model.py"),
    Path("Blackbox/TCN/train_tcn_model.py"),
    Path("TNN/train_tnn_model.py"),
]

def main():
    total_start = time.time()

    for i, script in enumerate(SCRIPTS, start=1):

        print("\n" + "=" * 70)
        print(f"Running {i}/{len(SCRIPTS)}: {script}")
        print("=" * 70)

        start = time.time()

        result = subprocess.run(
            [sys.executable, "-u", str(script)]
        )

        elapsed_h = (time.time() - start) / 3600

        if result.returncode != 0:
            print(
                f"\nFAILED: {script}\n"
                f"Exit code: {result.returncode}\n"
                f"Runtime: {elapsed_h:.2f} h"
            )
            return result.returncode

        print(
            f"\nFinished: {script}\n"
            f"Runtime: {elapsed_h:.2f} h"
        )

    total_h = (time.time() - total_start) / 3600

    print("\n" + "=" * 70)
    print("ALL HYPERPARAMETER SEARCHES FINISHED")
    print(f"Total runtime: {total_h:.2f} h")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())