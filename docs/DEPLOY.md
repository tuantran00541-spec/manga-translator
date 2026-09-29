# Deploying the Manga Cloud gateway

The app runs on each user's machine. Only the gateway in `gateway/` is hosted. It handles email login, plans, payments and the A.I calls of A.I mode. It is small: no models, no GPU, about 100 MB of RAM.

## What you need

- A Linux server with Docker and Docker Compose, ports 80 and 443 open.
- A domain name (for example `cloud.example.com`) whose DNS A record points at the server.
- API keys for the A.I upstream, Resend (login emails), and payOS and/or Lemon Squeezy.

## First start

```bash
git clone https://github.com/tuantran00541-spec/manga-translator.git
cd manga-translator/deploy
cp gateway.env.example gateway.env    # fill in the keys; git ignores this file
chmod 600 gateway.env
GATEWAY_DOMAIN=cloud.example.com docker compose up -d --build
curl https://cloud.example.com/health
```

Caddy gets the HTTPS certificate on its own. The gateway itself is not published on any port; only Caddy reaches it. It trusts the forwarded client address from Caddy's fixed network address alone, so the per-network limits cannot be fooled with a fake `X-Forwarded-For` header.

Then, in the payment dashboards, set the webhooks:

| Provider | Webhook URL |
| --- | --- |
| payOS | `https://cloud.example.com/v1/billing/payos/webhook` |
| Lemon Squeezy | `https://cloud.example.com/v1/billing/lemonsqueezy/webhook` (subscription events, signing secret = `GATEWAY_LS_WEBHOOK_SECRET`) |

Users point their app at the gateway with `MANGA_TIERS=1` and `MANGA_CLOUD_URL=https://cloud.example.com/v1`.

## Limits the gateway enforces

| Limit | Value |
| --- | --- |
| Chapters of A.I mode per month | Free 3, Plus 30, Pro 100 |
| Most one chapter may spend | Free $0.75, paid $2 (a long webtoon chapter costs about $0.25) |
| A.I calls per chapter | 1000 (a long chapter makes about 280) |
| Login codes | 1 a minute and 10 a day per email; 30 a day per network address |
| New accounts | 3 a day per network address |
| Request body | 32 MB, at most 24 inline images, only chat fields the app uses |

A deleted account keeps a hash of its email and its usage, so deleting it and signing up again does not reset the free quota. The email address itself is gone.

## Operating it

```bash
# Look up a user, see totals, change a plan by hand.
curl -H "X-Admin-Key: $KEY" "https://cloud.example.com/v1/admin/accounts?email=user@example.com"
curl -H "X-Admin-Key: $KEY" https://cloud.example.com/v1/admin/stats
curl -H "X-Admin-Key: $KEY" -H 'Content-Type: application/json' \
     -d '{"plan": "plus", "days": 30}' https://cloud.example.com/v1/admin/accounts/<account_id>/plan

# Back up the database while the gateway runs (daily from cron is enough).
docker compose exec gateway python -m gateway backup /data/backup-$(date +%F).sqlite
docker compose cp gateway:/data/backup-$(date +%F).sqlite ./

# Update.
git pull && GATEWAY_DOMAIN=cloud.example.com docker compose up -d --build
```

Expired sessions, spent login codes, old counters and stale jobs are cleaned every six hours by the gateway itself.

## Before going live

- `GATEWAY_DEV_LOGIN` must be unset: with it, login codes are returned in the API reply.
- `GATEWAY_ADMIN_KEY` must be long and random; without it the admin API is closed.
- Set a spending limit at the A.I upstream too, as a last line behind the per-chapter guard.
- Keep backups off the server.
