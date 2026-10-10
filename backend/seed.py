"""Dev-only demo accounts: one per real dataset recording, preloaded with its 5-minute history.

Each account is named patient<id> (e.g. 2072_0_20210706 -> patient2072). Histories are only
loaded into accounts that have none, so uploads and saved forecasts survive restarts.
"""
from pathlib import Path

import pandas as pd

from .ml.export_patient import to_five_minute
from .ml.service import COLUMNS, validate_history


def username_for(recording):
    return "patient" + recording.split("_")[0]


def seed_patients(accounts, data_dir, recordings):
    """recordings: [(recording, password)]. A changed password re-keys the account and ends its sessions."""
    seeded = []
    for recording, password in recordings:
        source = next(Path(data_dir).glob(recording + ".xls*"), None)
        if source is None:
            raise FileNotFoundError(f"No workbook named {recording}.xlsx in {data_dir}")
        user_id = accounts.ensure_user(username_for(recording), password)
        saved, revision = accounts.history(user_id)
        if not saved:
            raw = to_five_minute(pd.read_excel(source))
            for col in COLUMNS:
                if col not in raw:
                    raw[col] = pd.NA
            accounts.save_history(user_id, validate_history(raw[COLUMNS]).to_csv(index=False), revision)
        seeded.append(username_for(recording))
    return seeded
