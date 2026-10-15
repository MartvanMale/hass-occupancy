# Running without Supervisor

Use this if you run Home Assistant Container or Core and have no add-on store.
It is the same app as the add-on, packaged as a Docker image. It reads your
Home Assistant with a long-lived access token and publishes its sensors to your
MQTT broker.

## What you need

- A long-lived access token. Create one in Home Assistant under your profile, on the Security tab.
- An MQTT broker, with the MQTT integration set up in Home Assistant. Without it the sensors do not show up.
- An `amd64` or `aarch64` machine to run the image on.

## Setting it up

1. Make a folder for it, and put this `compose.yaml` in it:

   ```yaml
   services:
     occupancy-forecast:
       image: ghcr.io/martvanmale/occupancy-forecast:latest
       restart: unless-stopped
       ports:
         - "${LAN_ADDRESS}:8099:8099"
       volumes:
         - ./occupancy-data:/data
       env_file: .env
       healthcheck:
         test: ["CMD", "python", "-c", "import ssl, urllib.request; c = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT); c.check_hostname = False; c.verify_mode = ssl.CERT_NONE; urllib.request.urlopen('https://127.0.0.1:8099/health', context=c)"]
         interval: 1m
         timeout: 10s
         start_period: 2m
   ```

2. Put your settings in a file called `.env` in the same folder:

   ```sh
   LAN_ADDRESS=192.0.2.10
   HA_URL=http://homeassistant.local:8123
   HA_TOKEN=<long-lived access token>
   OCCUPANCY_PASSWORD=<a password for the panel>
   MQTT_HOST=<broker host>
   MQTT_USER=<broker user>
   MQTT_PASSWORD=<broker password>
   ```

   `LAN_ADDRESS` is the address of this machine on your network. The file holds your token and passwords, so make it readable only by you with `chmod 600 .env`, and keep it out of git.

3. Start it with `docker compose up -d`.
4. Open `https://<LAN_ADDRESS>:8099`. The first time, your browser warns that the certificate is not trusted, because the container made it itself. Accept it, then log in with any user name and your `OCCUPANCY_PASSWORD`.
5. Follow the [setup guide](../occupancy-forecast/DOCS.md#setting-up). You can also enter the broker on the panel's Connections tab. The `MQTT_*` settings are then no longer used.

Leave out `init: true` (or `--init` with `docker run`). The container stops
straight away if you set it.

## Settings

| Variable | What it does |
|---|---|
| `HA_URL` | Required. The address of your Home Assistant, for example `http://homeassistant.local:8123`. |
| `HA_TOKEN` | Required. Your long-lived access token. |
| `OCCUPANCY_PASSWORD` | Required. The password for the panel and the API. Any user name works. |
| `MQTT_HOST`, `MQTT_PORT`, `MQTT_USER`, `MQTT_PASSWORD`, `MQTT_SSL` | Your broker, if you do not enter it on the panel. The port defaults to 1883. Set `MQTT_SSL` to `true` for a TLS broker. |
| `OCCUPANCY_TOPIC_PREFIX` | The name the sensors are published under. Defaults to `occupancy_forecast`. Use lowercase letters, digits and underscores. Change it if another copy already publishes to the same broker. |
| `OCCUPANCY_SSL_CERT`, `OCCUPANCY_SSL_KEY` | Your own certificate and key, as paths inside the container. Leave both out to use the certificate the container makes. See [Your own certificate](#your-own-certificate). |
| `LOG_LEVEL` | `debug`, `info`, `warning` or `error`. Defaults to `info`. |

The time zone and country are taken from Home Assistant.

## Security

The panel and the API are served over HTTPS, so the password is encrypted on
the network. On first start the container makes a self-signed certificate and
stores it in `occupancy-data/tls`. It covers `localhost` and `LAN_ADDRESS`. If
you change `LAN_ADDRESS`, delete `occupancy-data/tls` and restart the container
to get a new one.

`/health` works without the password. It shows the status and the version, and
nothing about your household.

## Your own certificate

To stop the browser warning, use a certificate your devices trust, for example
one from your reverse proxy or from Let's Encrypt.

1. Put the certificate and its key in a folder next to `compose.yaml`, for example `certs`.
2. Add the folder to `volumes:` in `compose.yaml`:

   ```yaml
       volumes:
         - ./occupancy-data:/data
         - ./certs:/certs:ro
   ```

3. Add the paths to `.env`:

   ```sh
   OCCUPANCY_SSL_CERT=/certs/fullchain.pem
   OCCUPANCY_SSL_KEY=/certs/privkey.pem
   ```

4. Restart the container. It reads the files on every start, so a renewed certificate is picked up at the next restart.

## Updating the forecast on demand

The forecast already updates by itself when someone arrives, leaves or changes
zone. To trigger it yourself, for example from an automation, add a
`rest_command`:

```yaml
rest_command:
  occupancy_forecast_predict:
    url: https://192.0.2.10:8099/predict
    method: post
    verify_ssl: false
    username: forecast
    password: !secret occupancy_password
```

With your own certificate, leave out `verify_ssl: false`.

## Your data

Everything is stored in the `occupancy-data` folder next to `compose.yaml`: the
presence history, your settings and the trained models. Back up that folder and
`.env`, and you have everything.
