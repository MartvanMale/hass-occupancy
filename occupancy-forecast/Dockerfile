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

# A venv, because Debian marks its system Python externally-managed (PEP 668).
# libgomp1 is what scikit-learn's threaded estimators link against -- without it
# the failure is an obscure import error.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-venv libgomp1 \
 && python3 -m venv "$VIRTUAL_ENV" \
 && rm -rf /var/lib/apt/lists/*

# The account the forecaster runs as. No `USER` directive on purpose: s6-overlay
# is PID 1 and must start as root, so the drop is at the last exec in run.sh.
# The image stays root-owned and therefore read-only to the process.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin occupancy

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The built panel, committed rather than compiled here -- DEVELOPMENT.md ("The
# panel") says why. Before the Python, so the panel layer stays cached across ordinary edits.
COPY panel/dist ./occupancy_forecast/web/dist/

COPY occupancy_forecast/ ./occupancy_forecast/

COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]
