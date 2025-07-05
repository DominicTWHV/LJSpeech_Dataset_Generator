if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root"
    exit
fi
if command -v docker &> /dev/null
then
    echo "Docker is already installed"
else
    echo "Docker could not be found, installing..."
    
    apt update
    apt install -y docker.io
fi

docker build -t LJSpeech_DSG .

docker run LJSpeech_DSG