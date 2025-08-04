#!/bin/bash

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <service_name> [command]"
    echo "Example: $0 gemma3-1b install"
    exit 1
fi

# Build JSON args array from command line arguments
json_args=""
for arg in "$@"; do
    if [ -z "$json_args" ]; then
        json_args="\"$arg\""
    else
        json_args="$json_args, \"$arg\""
    fi
done

# Execute curl command with the arguments
curl -v -X POST http://localhost:6543/admin \
    -H "Content-Type: application/json" \
    -d "{\"args\": [$json_args]}"
