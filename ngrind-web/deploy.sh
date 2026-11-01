#!/bin/sh
set -e
cd "$(dirname "$0")/.."
./ngrind-web/build.sh
# Pin wrangler so deploys do not silently pick up a new (or compromised)
# release; bump the version deliberately.
bunx wrangler@4 pages deploy target/web --project-name ngrind --branch main --commit-dirty=true
