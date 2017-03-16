#!/bin/sh
ffmpeg -framerate 25 -i images/img_%04d.png -c:v libx264 -preset slow -crf 20 out.mp4
