#!/bin/sh
set -e
cd "$(dirname "$0")/.."
./ngrind-web/build.sh
bunx wrangler pages deploy target/web --project-name ngrind --branch main --commit-dirty=true
