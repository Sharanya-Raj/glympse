
# pyrefly: ignore [missing-import]
from mangum import Mangum
from backend.app import app

handler = Mangum(app)