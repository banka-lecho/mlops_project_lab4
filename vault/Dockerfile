FROM hashicorp/vault:1.20

COPY seed.sh /seed.sh
COPY entrypoint.sh /usr/local/bin/vault-entrypoint.sh
RUN chmod +x /usr/local/bin/vault-entrypoint.sh /seed.sh

EXPOSE 8200
ENTRYPOINT ["/usr/local/bin/vault-entrypoint.sh"]
