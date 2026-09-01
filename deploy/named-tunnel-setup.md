# Giving the bureau a stable hostname

The quick tunnel currently in front of the bureau proves the path works but is
not an endpoint: its hostname changes on every restart. A named tunnel keeps
it, which is the difference between a demo and something a partner configures
once and forgets.

Two ways to do this. The first is better for a headless box.

## A. Dashboard-managed tunnel (recommended)

No browser flow on the server, no `cert.pem`, no credentials JSON. Cloudflare
creates the DNS record for you. One token, which lives in one file.

### In the Cloudflare dashboard

1. **Zero Trust -> Networks -> Tunnels -> Create a tunnel -> Cloudflared.**
   Name it `freeboard-bureau`.
2. Cloudflare shows an install command containing a long token. **Copy only the
   token** -- the part after `--token`. Ignore the rest of the command; the
   service below runs it properly.
3. Still in the dashboard, add a **Public Hostname**:
   - Subdomain `bureau`, Domain `freeboardrisk.com`
   - Type **HTTP**, URL **127.0.0.1:8080**
   That creates the DNS record automatically. Nothing to add by hand.

### On the box

```bash
sudo mkdir -p /etc/cloudflared
sudo install -m 600 -o root -g root /dev/null /etc/cloudflared/token.env
sudo tee /etc/cloudflared/token.env >/dev/null <<'ENV'
TUNNEL_TOKEN=paste-the-token-here
ENV
sudo chmod 600 /etc/cloudflared/token.env
sudo chown sentinel:sentinel /etc/cloudflared/token.env
```

Then say so, and the service swap and verification can be done for you.

**Do not** paste the token into a chat, a commit, or a command line. A token in
a transcript is compromised from the moment it is sent, and this one authorises
running a tunnel on your account. `EnvironmentFile` keeps it out of `ps` too.

## B. Locally-managed tunnel (browser login)

Only if you specifically want the tunnel defined on the box rather than in the
dashboard.

```bash
cloudflared tunnel login          # prints a URL; open it in your own browser
cloudflared tunnel create freeboard-bureau
cloudflared tunnel route dns freeboard-bureau bureau.freeboardrisk.com
```

Then install `cloudflared-config.example.yml` as `/etc/cloudflared/config.yml`,
substitute the tunnel UUID and hostname, copy the credentials JSON beside it at
mode 600, and use `cloudflared-named.service` instead.

Note the login runs **on the box**, not on your laptop -- the credential has to
end up where the tunnel runs. It prints a URL you open in any browser.

## Afterwards

```bash
curl https://bureau.freeboardrisk.com/v1/health
```

The quick tunnel (`cloudflared-bureau.service`) gets disabled in the swap. Keep
it installed but stopped: it costs nothing and is a fallback if the named
tunnel is ever misconfigured.
