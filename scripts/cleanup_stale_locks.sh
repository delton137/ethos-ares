#!/bin/bash

# Script to clean up stale lock files from previous tokenization runs
# This helps resolve the "Waiting for workers" issue

echo "Cleaning up stale lock files..."

# Remove lock files older than 1 day
echo "Removing lock files older than 1 day..."
find . -name "*.json" -path "*/.data_*.parquet_cache/locks/*" -mtime +1 -delete 2>/dev/null || true

# Remove empty lock directories
echo "Removing empty lock directories..."
find . -type d -name "locks" -empty -delete 2>/dev/null || true

# Remove empty cache directories
echo "Removing empty cache directories..."
find . -type d -name ".data_*.parquet_cache" -empty -delete 2>/dev/null || true

echo "Cleanup completed!"
echo ""
echo "You can now run your tokenization process without the lock file issues."

