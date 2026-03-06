#!/bin/bash

# One-click script to pull latest code from GitHub
# Usage: ./git_pull.sh

# Ensure we're on the main branch
git checkout master

# Pull latest updates from remote main branch
git reset --hard
git pull origin master

echo "✅ Successfully pulled latest code from remote repository (master branch)"
