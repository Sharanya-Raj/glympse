# Email sign-in and password recovery

The sign-in form now accepts email and password only. `POST /sessions` takes
`{"email":"person@example.com","password":"..."}`. `/me` returns `email`;
authorization and patient history remain keyed by the verified Cognito `sub`.
The sign-in page has no account-confirmation field and signup remains off by default.

Forgot password uses `POST /password/forgot` with `email`. The separate recovery
screen submits `email`, `code`, and `password` to `POST /password/reset`.
It supports resend, repeated-password checking, errors, and return to sign-in.
Cognito's password policy is authoritative. Passwords/reset codes are never saved
in application storage. Unknown or recovery-ineligible accounts receive the same
initial message; incorrect/expired codes, throttling, and service failures surface
as errors. The optional existing signup/confirmation API remains separate from login.

The configured Cognito client must permit `ALLOW_USER_PASSWORD_AUTH` and have no
client secret. The existing user must be enabled and confirmed, with a permanent
password and the supplied email as its username or supported email alias. For
email recovery, it needs a verified email and a pool recovery policy that allows
email. MFA/temporary-password challenges still require the administrator's setup;
removing a field does not bypass Cognito challenges or verify an email address.
See [AWS password recovery](https://docs.aws.amazon.com/cognito/latest/developerguide/managing-users-passwords.html)
and [ForgotPassword](https://docs.aws.amazon.com/cognito-user-identity-pools/latest/APIReference/API_ForgotPassword.html).

Deploy frontend and backend together because the credentials contract changed
from `username` to `email`. API Gateway must forward `/password/forgot` and
`/password/reset`. No deployment or live AWS configuration change has been made.
Use the Cognito/table settings from `.env.example`; remove retired local admin
and dataset-seeding settings from local `.env` files before running the server.

Code verification includes email-only login, token/sub ownership, recovery,
resend, invalid/expired codes, password policies, failed requests, and signing in
with the changed password. Tests use mocked Cognito, not the real account.
Actual email delivery and live Cognito configuration await AWS access.

## Patient 2088

The current source is `data/T2DM_5min_with_macros.csv`, not the earlier prepared
validation dataset. `scripts/prepare_patient_csv.py` produces the cleaned,
app-compatible `data/sharanya_cg/patient_2088.csv`; see
[script usage](../scripts/PATIENT_CSV.md).

Once local AWS credentials are available, preview and then import:

```powershell
python scripts/import_participant_history.py --source data/sharanya_cg/patient_2088.csv --patient 2088 --email sharanyaraj24@gmail.com --profile YOUR_PROFILE --dry-run
python scripts/import_participant_history.py --source data/sharanya_cg/patient_2088.csv --patient 2088 --email sharanyaraj24@gmail.com --profile YOUR_PROFILE --write
```

Omit `--profile` for the default credential chain. The script resolves the email's
Cognito `sub` using `AdminGetUser`; optional `--expected-sub` adds a known-ID check.
Defaults use the repository's pool `us-west-2_xjNDVpyKd` and region `us-west-2`.
Review/override `--user-pool-id` and `--region` if the account uses another pool.
It needs Cognito `AdminGetUser` and DynamoDB `GetItem`, `Query`, `PutItem`, and
`UpdateItem` on the configured pool/tables. Identical imports are no-ops, and
different existing history is refused. No password or access key is needed in chat.
