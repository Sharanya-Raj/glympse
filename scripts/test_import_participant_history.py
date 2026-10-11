import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import boto3
from moto import mock_aws
from backend.accounts import Accounts
from scripts.import_participant_history import select_history, import_history, resolve_identity


class ImportTests(unittest.TestCase):
    def test_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.csv'
            source = 'Patient,Recording,Date,CGM (mg / dl),Dietary intake,Interpolated,Extra\n2000,r,2025-01-01 00:00,110.00,Rice 100 g,FALSE,NA\n2000,r,2025-01-01 00:05,111.50,,TRUE,original\n2001,s,2025-01-01 00:00,120,,FALSE,other\n'
            path.write_text(source, encoding='utf-8')
            csv, report = select_history(path, '2000')
            self.assertEqual(csv, '\n'.join(source.splitlines()[:3]) + '\n')
            self.assertEqual(report['food_event_rows'], 1)
            self.assertEqual(report['source_interpolated_rows'], 1)
            with self.assertRaises(ValueError):
                select_history(path, '1')

    def test_identity_guard(self):
        client = Mock()
        client.admin_get_user.return_value = {'Enabled': True, 'UserAttributes': [{'Name':'sub','Value':'synthetic-a'}]}
        self.assertEqual(resolve_identity(client, 'patient1', 'synthetic-a'), 'synthetic-a')
        with self.assertRaises(ValueError):
            resolve_identity(client, 'patient1', 'synthetic-b')

    @mock_aws
    @patch.dict('os.environ', {'GLYMPSE_PATIENTS_TABLE':'glympse-patients','GLYMPSE_MEALS_TABLE':'glympse-meals'})
    def test_dry_run_idempotence_and_isolation(self):
        resource = boto3.resource('dynamodb', region_name='us-west-2')
        for name, keys in [('glympse-patients',['patient_id']),('glympse-meals',['patient_id','meal_id'])]:
            resource.create_table(TableName=name, KeySchema=[{'AttributeName':k,'KeyType':'HASH' if i == 0 else 'RANGE'} for i,k in enumerate(keys)], AttributeDefinitions=[{'AttributeName':k,'AttributeType':'S'} for k in keys], BillingMode='PAY_PER_REQUEST')
        accounts = Accounts(resource)
        self.assertFalse(import_history(accounts, 'synthetic-a','patient1','original\n')['written'])
        self.assertEqual(accounts.patient('synthetic-a'), {})
        self.assertTrue(import_history(accounts,'synthetic-a','patient1','original\n',True)['written'])
        chunks = list(accounts.query('synthetic-a','history#'))
        self.assertFalse(import_history(accounts,'synthetic-a','patient1','original\n',True)['written'])
        self.assertEqual(list(accounts.query('synthetic-a','history#')), chunks)
        self.assertEqual(accounts.history('synthetic-a'), ('original\n',1))
        self.assertEqual(accounts.history('synthetic-b'), (None,0))
        with self.assertRaises(ValueError):
            import_history(accounts,'synthetic-a','patient1','replacement\n',True)
        self.assertEqual(accounts.history('synthetic-a'), ('original\n',1))


if __name__ == '__main__':
    unittest.main()
