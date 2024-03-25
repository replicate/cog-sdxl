#!/bin/bash

# Check if at least one URL is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <URL1> <URL2> ..."
    exit 1
fi

# Create the manifest content
manifest_content=""
for url in "$@"; do
    # skip -s and -v
    if [ "$url" = "-s" ] || [ "$url" = "-v" ]; then
        continue
    fi
    manifest_content+="$url stdout\n"
done
# strip final newline
manifest_content=$(echo -e "$manifest_content" | head -n -1)

# Execute pget with the manifest passed through stdin

echo -e "$manifest_content" | /src/pget --chunk-size=250M --concurrency=5 --pid-file $(mktemp) multifile -o stdout -
