import os

import hvac


class VaultClient:
    def __init__(
        self,
        url: str = None,
        token: str = None,
    ):
        self.url = url or os.getenv(
            "VAULT_ADDR",
            "http://localhost:8200",
        )

        self.token = token or os.getenv("VAULT_TOKEN")

        if not self.token:
            raise RuntimeError("VAULT_TOKEN is not set")

        self.client = hvac.Client(
            url=self.url,
            token=self.token,
        )

        if not self.client.is_authenticated():
            raise RuntimeError("Vault authentication failed")

    def get_secrets(self, path: str) -> dict:
        response = self.client.secrets.kv.v2.read_secret_version(
            mount_point="secret",
            path=path,
        )

        return response["data"]["data"]
