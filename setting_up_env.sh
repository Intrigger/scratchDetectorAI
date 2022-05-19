echo "Starting installation!"

python3 -m venv venv;
source venv/bin/activate;
pip install tensorflow;
pip install matplotlib;
pip install Pillow

echo "Installation Finished!"
