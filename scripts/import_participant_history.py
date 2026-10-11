"""One-time original-history import. Dry-run by default; no model loading/training."""
import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.accounts import Accounts
from backend.ml.service import validate_history

REGION = 'us-west-2'
POOL = 'us-west-2_xjNDVpyKd'


def select_history(source, patient, recording=None, exclude_columns=(), exclude_chinese_columns=False):
    # String loading retains original precision, text, empty cells and NA literals.
    data = pd.read_csv(source, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if 'Patient' not in data:
        raise ValueError('Source must contain an explicit Patient column')
    selected = data.loc[data.Patient.eq(str(patient))].copy()
    if recording is not None:
        selected = selected.loc[selected.Recording.eq(recording)].copy()
    if selected.empty:
        raise ValueError(f'No rows for Patient {patient}; available IDs: {list(data.Patient.unique())[:30]}')
    if selected.Recording.nunique() != 1:
        raise ValueError('Select exactly one recording with --recording')
    excluded = set(exclude_columns)
    if missing := excluded - set(selected.columns):
        raise ValueError(f'Excluded columns not found: {sorted(missing)}')
    if exclude_chinese_columns:
        excluded.update(c for c in selected.columns if re.search(r'[\u3400-\u9fff\uf900-\ufaff]', c))
    if excluded & {'Patient', 'Recording', 'Date', 'CGM (mg / dl)'}:
        raise ValueError('Cannot exclude patient, recording, timestamp or CGM columns')
    selected = selected.drop(columns=sorted(excluded))
    # Validate exactly as the upload endpoint, but archive ALL original columns and
    # lexical values. The backend applies its original normalization on reads.
    validate_history(selected.replace('', pd.NA))
    csv = selected.to_csv(index=False, lineterminator='\n')
    food = selected['Dietary intake'].str.strip() if 'Dietary intake' in selected else pd.Series('', index=selected.index)
    meal = ~food.str.casefold().isin(['', 'no food ingested', 'none', 'nan', 'na', 'n/a'])
    report = {
        'excluded_columns': sorted(excluded),
        'source': str(Path(source).resolve()),
        'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'patient_in_source': str(patient), 'recordings': list(selected.Recording.unique()),
        'rows': len(selected), 'glucose_values': int(selected['CGM (mg / dl)'].ne('').sum()),
        'nonempty_food_descriptions': int(food.ne('').sum()), 'food_event_rows': int(meal.sum()),
        'first_timestamp': str(selected.Date.min()), 'last_timestamp': str(selected.Date.max()),
        'preserved_columns': list(selected.columns),
        'history_sha256': hashlib.sha256(csv.encode()).hexdigest(),
        'timestamp_policy': 'Original strings preserved; backend treats naive times as UTC without shifting the clock',
        'writes': 'CSV history chunks and patient history manifest only; no invented forecasts or meal effects',
    }
    if 'Interpolated' in selected:
        report['source_interpolated_rows'] = int(selected.Interpolated.str.lower().eq('true').sum())
    return csv, report


def resolve_identity(client, username, expected_sub=None, pool=POOL):
    result = client.admin_get_user(UserPoolId=pool, Username=username)
    sub = next((a['Value'] for a in result['UserAttributes'] if a['Name'] == 'sub'), None)
    if not sub or (expected_sub is not None and sub != expected_sub):
        raise ValueError('Cognito sub does not match the expected target; no writes performed')
    if not result.get('Enabled', False):
        raise ValueError('Target Cognito user is disabled')
    return sub


def import_history(accounts, sub, username, csv, write=False):
    existing, revision = accounts.history(sub)
    if existing == csv:
        return {'action': 'already_imported', 'revision': revision, 'written': False}
    if existing is not None:
        raise ValueError('Target already has different history. Refusing to overwrite or merge it.')
    if not write:
        return {'action': 'would_import', 'revision': revision, 'written': False}
    # Conditional creation does not overwrite an existing account or its metadata.
    try:
        accounts.patients.put_item(Item={'patient_id': sub, 'username': username, 'revision': 0, 'epoch': 0},
                                  ConditionExpression='attribute_not_exists(patient_id)')
    except ClientError as exc:
        if exc.response['Error']['Code'] != 'ConditionalCheckFailedException':
            raise
    # Recheck after creation to prevent overwriting a concurrent upload.
    existing, current_revision = accounts.history(sub)
    if existing == csv:
        return {'action': 'already_imported', 'revision': current_revision, 'written': False}
    if existing is not None or current_revision != revision:
        raise ValueError('Target changed during import; refusing to overwrite')
    accounts.save_history(sub, csv, revision)
    saved, next_revision = accounts.history(sub)
    if saved != csv:
        raise RuntimeError('Post-write history verification failed; inspect concurrent uploads')
    return {'action': 'imported_and_verified', 'revision': next_revision, 'written': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--patient', required=True, help='Exact original Patient ID; never inferred from username')
    parser.add_argument('--recording', help='Required when the source patient has multiple recordings')
    parser.add_argument('--exclude-column', action='append', default=[], help='Column to omit; repeat for multiple columns')
    parser.add_argument('--exclude-chinese-columns', action='store_true', help='Omit columns with Chinese characters in their names')
    parser.add_argument('--output', type=Path, help='Also export the selected CSV locally without replacing different existing data')
    parser.add_argument('--email', '--username', dest='username', required=True, help='Exact Cognito email sign-in identifier')
    parser.add_argument('--expected-sub', help='Optional additional identity check against a known Cognito sub')
    parser.add_argument('--region', default=REGION)
    parser.add_argument('--user-pool-id', default=POOL)
    parser.add_argument('--profile', help='Existing AWS CLI/SSO profile; never supply keys in source code')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--dry-run', action='store_true', help='Default; Cognito/DynamoDB reads only')
    mode.add_argument('--offline', action='store_true', help='Dataset preview only; no AWS calls')
    args = parser.parse_args()
    csv, report = select_history(args.source, args.patient, args.recording, args.exclude_column, args.exclude_chinese_columns)
    if args.output:
        if args.output.resolve() == args.source.resolve():
            raise ValueError('Output cannot replace the source dataset')
        if args.output.exists() and args.output.read_bytes() != csv.encode('utf-8'):
            raise ValueError('Output already contains different data; choose another path')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(csv.encode('utf-8'))
        report['export'] = str(args.output.resolve())
    report.update(region=args.region, user_pool_id=args.user_pool_id, email=args.username, mode='write' if args.write else 'dry-run',
                  patients_table='glympse-patients', meals_table='glympse-meals')
    if args.offline:
        report.update(mode='offline', identity_verified=False, written=False)
    else:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        sub = resolve_identity(session.client('cognito-idp'), args.username, args.expected_sub, args.user_pool_id)
        accounts = Accounts(session.resource('dynamodb'))
        # Refuse table overrides to avoid a preview/write targeting unexpected tables.
        if accounts.patients.name != 'glympse-patients' or accounts.meals.name != 'glympse-meals':
            raise ValueError('Remove GLYMPSE_*_TABLE overrides before this one-time import')
        report.update(patient_id=sub, identity_verified=True)
        report.update(import_history(accounts, sub, args.username, csv, args.write))
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()

