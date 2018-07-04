#!/usr/bin/env bash
gunicorn app:app --bind 0.0.0.0:9666 -w 2 --worker-class aiohttp.GunicornWebWorker -t 180 -p pid.file --access-logfile=./log/access.log --log-level=info --log-file=./log/error.log &>./log/log.txt &
