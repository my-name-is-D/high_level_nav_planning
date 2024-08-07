#Install Ubuntu 20.04
FROM ubuntu:20.04

# Set environment variable to use host's X11 display
ENV DISPLAY=:0

# Install Xvfb and necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    x11-apps \   
    && apt-get clean

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.9 python3-tk libcairo2-dev

# Make sure everything is up to date before building from source
RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get clean \
  && apt-get -y install python3-pip \ 
  && pip install --upgrade pip
  # Install necessary packages for Python and GUI support

COPY . /higher_level_nav
RUN rm -r /higher_level_nav/results/* \
    && rm -r /higher_level_nav/figures

RUN cd higher_level_nav \ 
    && pip install .

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]


#command: docker build -t hstnav .
