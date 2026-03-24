#!/bin/bash
# entrypoint.sh - Runs any setup logic and executes the command

set -e

# Insert initialization logic here (e.g., config generation, setup verification)
echo "Starting environment for gnc-toolkit"

# Execute the main container command (e.g., pytest, bash)
exec "$@"
