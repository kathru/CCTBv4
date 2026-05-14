#!/bin/bash
# Get full version: vMAJOR.STRATEGY.BUILD
# BUILD = total commit count

VERSION=$(cat VERSION 2>/dev/null || echo "4.0")
BUILD=$(git rev-list --all --count 2>/dev/null || echo "0")
echo "v${VERSION}.${BUILD}"
