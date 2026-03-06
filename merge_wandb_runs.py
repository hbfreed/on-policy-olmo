"""One-off script to merge two wandb runs into a new combined run.

Usage:
    uv run python merge_wandb_runs.py

For each pair, fetches full (unsampled) history from both runs, combines them
sorted by step, and logs to a new run. The original runs are left untouched.
"""

import wandb
import pandas as pd

PROJECT = "hbfreed/olmo-2-1b-off-policy-distillation"
ENTITY = "hbfreed"

# (first/original run, second/resumed run)
PAIRS = [
    ("xhzvc6kp", "4lgfdhqk"),
    ("0s8tahku", "71esqrq5"),
]

api = wandb.Api()

for first_id, second_id in PAIRS:
    print(f"\n=== Combining {first_id} + {second_id} ===")

    run1 = api.run(f"{PROJECT}/{first_id}")
    run2 = api.run(f"{PROJECT}/{second_id}")

    # Use scan_history() for full unsampled data
    h1 = pd.DataFrame(list(run1.scan_history()))
    h2 = pd.DataFrame(list(run2.scan_history()))

    print(f"Run {first_id}: {len(h1)} rows, steps {h1['_step'].min()}-{h1['_step'].max()}")
    print(f"Run {second_id}: {len(h2)} rows, steps {h2['_step'].min()}-{h2['_step'].max()}")

    # Combine: use run1 for overlapping steps (it's the original)
    combined = pd.concat([h1, h2]).drop_duplicates(subset=["_step"], keep="first")
    combined = combined.sort_values("_step").reset_index(drop=True)
    print(f"Combined: {len(combined)} rows, steps {combined['_step'].min()}-{combined['_step'].max()}")

    # Create new run with the first run's name and config
    new_run = wandb.init(
        project=PROJECT.split("/")[1],
        entity=ENTITY,
        name=f"{run1.name} (merged)",
        config=run1.config,
    )

    for _, row in combined.iterrows():
        data = row.dropna().to_dict()
        step = int(data.pop("_step", 0))
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        new_run.log(data, step=step)

    new_run.finish()
    print(f"Created new merged run: {new_run.id}")
    print(f"You can now archive/delete {first_id} and {second_id} from the wandb UI.")

print("\nAll merges complete!")
