@echo off
setlocal enabledelayedexpansion

for %%a in (*.jpg) do (
  ffmpeg -i "%%a" -vf "scale=256:256" -y "temp.jpg"
  del "%%a"
  ren "temp.jpg" "%%~na.jpg"
)

echo All images resized to 256x256.
