# Giving the bureau a stable hostname

The quick tunnel currently in front of the bureau proves the path works, but it
is not an endpoint: its hostname changes on every restart and Cloudflare offer
no uptime guarantee. A design partner needs something they configure once.

Two prerequisites are yours and cannot be automated:

1. **A domain on Cloudflare.** A named tunnel routes to a hostname inside a
   zone you control. `trycloudflare.com` is quick-tunnel-only, and Cloudflare
   no longer issues zones for domains you do not own. Register one (~$10/yr),
   then add it to Cloudflare and point the registrar's nameservers at the pair
   Cloudflare gives you. Propagation is usually under an hour.

2. **`cloudflared tunnel login`.** This opens a browser and authenticates as
   you. Run it yourself; it writes `~/.cloudflared/cert.pem` on the box, which
   is a credential and should never leave it.

## On the box, after those two

```bash
# 1. authenticate (browser opens; pick the zone you just added)
cloudflared tunnel login

# 2. create the tunnel and note the UUID it prints
cloudflared tunnel create freeboard-bureau

# 3. route a hostname to it
cloudflared tunnel route dns freeboard-bureau bureau.yourdomain.com

# 4. install config and credentials
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/<UUID>.json /etc/cloudflared/
sudo cp /opt/freeboard/deploy/cloudflared-config.example.yml /etc/cloudflared/config.yml
sudo sed -i "s/REPLACE_WITH_TUNNEL_ID/<UUID>/g;s/REPLACE_WITH_HOSTNAME/bureau.yourdomain.com/" \
     /etc/cloudflared/config.yml
sudo chown -R sentinel:sentinel /etc/cloudflared
sudo chmod 600 /etc/cloudflared/<UUID>.json

# 5. swap the quick tunnel for the named one
sudo systemctl disable --now cloudflared-bureau
sudo cp /opt/freeboard/deploy/cloudflared-named.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared-named

# 6. verify from somewhere else entirely
curl https://bureau.yourdomain.com/v1/health
```

## The alternative, if you would rather not buy a domain

Restore billing on the GCP project, then open 80/443 and use
`deploy/caddy.example`. Caddy obtains and renews Let's Encrypt certificates
itself. This still needs a domain for the certificate, so it is not cheaper --
it is only different. The tunnel wins on having no inbound attack surface and
on surviving the ephemeral IP, which the closed billing account makes
unavoidable anyway.

## What not to do

Do not put the credentials JSON or `cert.pem` in the repo, in an image, or in
an environment variable. They authorise creating tunnels on your account. Mode
600, owned by the service user, on the host only.
