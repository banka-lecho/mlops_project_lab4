import json
import os
import urllib.request

addr = os.environ["VAULT_ADDR"]
tok = os.environ["VAULT_TOKEN"]
path = os.environ.get("VAULT_SECRET_PATH", "mlops-app")
req = urllib.request.Request(
    f"{addr}/v1/secret/data/{path}", headers={"X-Vault-Token": tok}
)
data = json.load(urllib.request.urlopen(req))["data"]["data"]
for k, v in data.items():
    print(f"export {k}={json.dumps(str(v))}")
