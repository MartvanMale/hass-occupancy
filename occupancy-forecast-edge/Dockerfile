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

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY occupancy_forecast/ ./occupancy_forecast/

# The built Ingress panel, produced by scripts/build-panel.sh on the development
# machine and COMMITTED, so it is in the build context wherever that context
# comes from -- an rsync from a deploy script, or a git clone Supervisor made
# because somebody added this repository to their add-on store. Deliberately no
# node stage: the Home Assistant box builds this image, and no other add-on on it
# compiles a frontend locally.
#
# If this COPY fails, panel/dist is not in the tree. In a clone that means the
# bundle was committed stale or not at all, which is a bug here and not
# something the installing user can do anything about; on a development machine
# it means `scripts/build-panel.sh <tree>` has not been run.
COPY panel/dist ./occupancy_forecast/web/dist/

COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]
