#!/bin/bash

# Define the maximum batch size (1 GB in bytes)
MAX_SIZE=$((1 * 1024 * 1024 * 1024)) 
CURRENT_SIZE=0
BATCH=1

echo "Checking for individual files over 100MB (GitHub will block these)..."
find . -type f -size +99M
echo "If any files were listed above, press Ctrl+C now. Otherwise, starting push..."
sleep 5

# Loop through every file recursively
find . -type f ! -name 'chunk_push.sh' ! -path '*/.git/*' -print0 | while IFS= read -r -d '' file; do
    git add "$file"
    FILE_SIZE=$(stat -c%s "$file")
    CURRENT_SIZE=$((CURRENT_SIZE + FILE_SIZE))

    # When 1GB is reached, commit and push
    if [ $CURRENT_SIZE -ge $MAX_SIZE ]; then
        echo "Pushing Batch $BATCH (~1GB)..."
        git commit -m "Adding dataset batch $BATCH"
        git push -u origin main
        
        # Reset counters for the next batch
        CURRENT_SIZE=0
        BATCH=$((BATCH + 1))
    fi
done

# Push any remaining files that didn't reach the 1GB threshold
if [ $CURRENT_SIZE -gt 0 ]; then
    echo "Pushing final batch..."
    git commit -m "Adding final dataset batch"
    git push -u origin main
fi

echo "Upload complete!"
