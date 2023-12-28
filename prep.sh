#!/bin/sh
for i in audiofiles/*.mp3; do
    ffmpeg -i "$i" "wavfiles/$(basename "$i" .mp3).wav"
done;
