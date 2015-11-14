#!/usr/bin/env bash


########################################
# sync & update
sudo pacman -Sy --noconfirm
#sudo pacman -Syu --noconfirm


########################################
# utils
#sudo pacman -S --noconfirm wget net-tools


########################################
# libs
sudo pacman -S --noconfirm lapack gcc-fortran cmake


########################################
## git
#sudo pacman -S --noconfirm git
#git config --global user.name blxlrsmb
#git config --global user.email blxlrsmb


########################################
# python
sudo pacman -S --noconfirm python2 python2-pip
sudo pacman -S --noconfirm python2-numpy
sudo pip2 install scipy theano


########################################
# lua/torch
#   TODO: CUDA support
#curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
#git clone https://github.com/torch/distro.git ~/torch --recursive
#cd ~/torch; ./install.sh


########################################
# opencv
sudo pacman -S --noconfirm opencv
sudo ln /dev/null /dev/raw1394 # http://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394
