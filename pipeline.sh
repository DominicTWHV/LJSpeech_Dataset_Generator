#!/bin/bash

set -e

# system detection

detect_system() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                DISTRO="$ID"
            elif [ -f /etc/arch-release ]; then
                DISTRO="arch"
            else
                DISTRO="unknown"
            fi
            ;;
        Darwin)
            DISTRO="macos"
            ;;
        *)
            echo "[ERROR] Unsupported operating system: $OS"
            exit 1
            ;;
    esac

    echo "[OK] Detected: $OS ($ARCH) — $DISTRO"
}

# install deps

install_dependencies() {
    local INSTALLED=false

    case "$DISTRO" in
        ubuntu|debian|pop|linuxmint|elementary|raspbian)
            echo "[INFO] Using apt package manager"
            if ! command -v python3 &>/dev/null || \
               ! dpkg -s python3-venv &>/dev/null 2>&1 || \
               ! command -v pip3 &>/dev/null || \
               ! command -v ffmpeg &>/dev/null; then
                sudo apt update
                sudo apt install -y python3 python3-venv python3-pip ffmpeg
                INSTALLED=true
            fi
            ;;

        arch|manjaro|endeavouros|garuda)
            echo "[INFO] Using pacman package manager"
            local PKGS=""
            command -v python3 &>/dev/null || PKGS="$PKGS python"
            python3 -c "import venv" 2>/dev/null || PKGS="$PKGS python"
            command -v pip3 &>/dev/null || PKGS="$PKGS python-pip"
            command -v ffmpeg &>/dev/null || PKGS="$PKGS ffmpeg"
            if [ -n "$PKGS" ]; then
                sudo pacman -Syu --noconfirm $PKGS
                INSTALLED=true
            fi
            ;;

        fedora|rhel|centos|rocky|alma)
            echo "[INFO] Using dnf package manager"
            local PKGS=""
            command -v python3 &>/dev/null || PKGS="$PKGS python3"
            command -v pip3 &>/dev/null || PKGS="$PKGS python3-pip"
            command -v ffmpeg &>/dev/null || PKGS="$PKGS ffmpeg"
            if [ -n "$PKGS" ]; then
                sudo dnf install -y $PKGS
                INSTALLED=true
            fi
            ;;

        opensuse*|sles)
            echo "[INFO] Using zypper package manager"
            local PKGS=""
            command -v python3 &>/dev/null || PKGS="$PKGS python3"
            command -v pip3 &>/dev/null || PKGS="$PKGS python3-pip"
            command -v ffmpeg &>/dev/null || PKGS="$PKGS ffmpeg"
            if [ -n "$PKGS" ]; then
                sudo zypper install -y $PKGS
                INSTALLED=true
            fi
            ;;

        macos)
            echo "[INFO] Using Homebrew package manager"
            if ! command -v brew &>/dev/null; then
                echo "[ERROR] Homebrew is not installed. Install it from https://brew.sh"
                exit 1
            fi
            local PKGS=""
            command -v python3 &>/dev/null || PKGS="$PKGS python"
            command -v ffmpeg &>/dev/null || PKGS="$PKGS ffmpeg"
            if [ -n "$PKGS" ]; then
                brew install $PKGS
                INSTALLED=true
            fi
            ;;

        *)
            echo "[WARNING] Unknown distribution '$DISTRO'."
            echo "         Please install manually: python3, python3-venv, python3-pip, ffmpeg"
            echo "         Attempting to continue..."
            ;;
    esac

    if [ "$INSTALLED" = true ]; then
        echo
        echo -e "\033[35m=======================================================\033[0m"
        echo -e "[OK] \033[36mDependencies installed successfully.\033[0m"
        echo -e "\033[35m=======================================================\033[0m"
        echo
    else
        echo "[OK] Dependencies already installed."
    fi
}

# venv setup

setup_venv() {
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "[OK] Virtual environment already exists."
        source venv/bin/activate
    fi
}

# Main pipeline

mkdir -p wavs output


if [ "$(uname -s)" = "Linux" ]; then
    chown -R "$USER":"$USER" wavs output
fi
chmod -R 755 wavs output

detect_system
install_dependencies
setup_venv

echo
echo -e "\033[35m=======================================================\033[0m"
echo "Initializing the WebUI"
python3 webui.py

deactivate