ARG BUILD_FROM
FROM $BUILD_FROM

# The Home Assistant Debian base brings S6 and bashio but no Python, so it is
# installed here. See build.yaml for why this is not the Python base image.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# A venv rather than --break-system-packages: Debian marks its system Python
# externally-managed (PEP 668), and the venv is the sanctioned way round it
# rather than the override. It also keeps these pins from ever arguing with
# apt's own python3 packages.
#
# libgomp1 is what scikit-learn's threaded estimators link against at runtime;
# without it the failure is an obscure import error rather than a build error.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-venv libgomp1 \
 && python3 -m venv "$VIRTUAL_ENV" \
 && rm -rf /var/lib/apt/lists/*

# The account the forecaster actually runs as. There is no `USER` directive
# here on purpose: s6-overlay is PID 1 (config.yaml sets `init: false`) and has
# to start as root, so the drop happens at the last exec in run.sh instead.
#
# --system, no home, no shell: nothing about this account is meant to be logged
# into. It owns /data and nothing else -- the venv, /app and the rest of the
# image stay root-owned and therefore read-only to the process, which is half
# the point and the half apparmor.txt then enforces properly.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin occupancy

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The built Ingress panel, committed rather than compiled here -- DEVELOPMENT.md
# ("The panel") says why. If this COPY fails, run scripts/build-panel.sh.
#
# Before the Python, not after: the bundle changes far less often than the code,
# and Docker invalidates every layer below a changed one, so this order keeps the
# panel layer cached across ordinary Python edits.
COPY panel/dist ./occupancy_forecast/web/dist/

COPY occupancy_forecast/ ./occupancy_forecast/

COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]
