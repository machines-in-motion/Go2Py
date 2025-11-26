# System Setup

The GO2 EDU comes with an onboard Jetson Orin NX, a Hessai LX-16 LiDAR sensor, and an intel Realsense D435i camera. This setup procedure targets the onboard Jetson Orin with IP `192.168.123.18` and presents the instructions for installing the Go2Py. We use the Linux systemd infrastructure to automatically run the components of the Go2Py as the robot is turned on. Each service runs its particular docker image so the installation of these services involves two steps; first building the required images and then installing the service files that launch them. Before installing these services though, you would need to perform the following steps to get an internet connection on the robot and install docker on it. 

## Internet Sharing

In order to access the internet on the Jetson computer, we hook the robot to a host development computer with internet access and configure it to share its connection with the robot. To configure the host computer, the following steps should be taken:
### Host Computer
The following steps configure the host computer to share its internet with the robot.
#### Enable IP forwarding:

```bash
sudo sysctl -w net.ipv4.ip_forward=1
```
#### Configure the iptables:

```bash
sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
sudo iptables -A FORWARD -i wlan0 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A FORWARD -i eth0 -o wlan0 -j ACCEPT
```
Note that `wlan0` should be replaced with the actual name of the network interface over which the internet is provided to the host computer, and eth0 should be replaced with the name of the Ethernet interface connected to the robot and has an IP address in range `192.168.123.x`. 

#### Storing the Settings
Make the iptables rules persistent by installing the `iptables-persistent`:

```bash
sudo apt-get install iptables-persistent
sudo iptables-save > /etc/iptables/rules.v4
sudo ip6tables-save > /etc/iptables/rules.v6
```
### Configuring Apt Sources
By default, the unitree comes with jetpack revision `r35.1` and Ubuntu 20.04. If you're outside China, you can change the apt source lists to the default values for faster software download. First create a backup of the old apt sources:

```bash
sudo mv /etc/apt/sources.list /etc/apt/sources.list.old
```

Then copy the following into the `/etc/apt/sources.list`:

``` bash
# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.
deb http://ports.ubuntu.com/ubuntu-ports/ bionic main restricted
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic main restricted

## Major bug fix updates produced after the final release of the
## distribution.
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-updates main restricted
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic-updates main restricted

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team. Also, please note that software in universe WILL NOT receive any
## review or updates from the Ubuntu security team.
deb http://ports.ubuntu.com/ubuntu-ports/ bionic universe
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic universe
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-updates universe
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic-updates universe

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team, and may not be under a free licence. Please satisfy yourself as to
## your rights to use the software. Also, please note that software in
## multiverse WILL NOT receive any review or updates from the Ubuntu
## security team.
deb http://ports.ubuntu.com/ubuntu-ports/ bionic multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-updates multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic-updates multiverse

## N.B. software from this repository may not have been tested as
## extensively as that contained in the main release, although it includes
## newer versions of some applications which may provide useful features.
## Also, please note that software in backports WILL NOT receive any review
## or updates from the Ubuntu security team.
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-backports main restricted universe multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic-backports main restricted universe multiverse

## Uncomment the following two lines to add software from Canonical's
## 'partner' repository.
## This software is not part of Ubuntu, but is offered by Canonical and the
## respective vendors as a service to Ubuntu users.
# deb http://archive.canonical.com/ubuntu bionic partner
# deb-src http://archive.canonical.com/ubuntu bionic partner

deb http://ports.ubuntu.com/ubuntu-ports/ bionic-security main restricted
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic-security main restricted
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-security universe
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic-security universe
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-security multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports/ bionic-security multiverse
```
We can also enable the Nvidia mirrors as explained [here](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/SoftwarePackagesAndTheUpdateMechanism.html). Specifically, do `sudo vi /etc/apt/sources.list.d/nvidia-l4t-apt-source.list` and copy the following inside:
```bash
deb https://repo.download.nvidia.com/jetson/common r35.1 main
deb https://repo.download.nvidia.com/jetson/t234 r35.1 main
```
Finally, update the and upgrade:
```bash
sudo apt update
sudo apt upgrade
```
### Installing CUDA

To use the onboard GPU we need to install the CUDA toolkit for Jetson [here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=20.04&target_type=deb_local). Specifically:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
sudo dpkg -i cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
sudo cp /var/cuda-tegra-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-11-8
```
Additionally, we also need to install the following packages for successful CUDA kernel compilation in our projects:
```
sudo apt install -y libcudnn8-dev libcusolver-dev-11-8 libcublas-dev-11-8 libcublas-11-8 libcusparse-11-8 libcusparse-dev-11-8
```
In order to install Torch, we need to know what Jetpack revision we have. You can check it out through the following command:
```
dpkg-query --show nvidia-l4t-core
```
### Installing Conda

Simply run the following commands to install Conda:

```bash
sudo chown $USER /opt
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O ~/miniconda.sh
/bin/bash ~/miniconda.sh -b -p /opt/conda 
rm ~/miniconda.sh 
sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh 
echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
echo "conda activate base" >> ~/.bashrc
```

Now create a virtual environment for the deployment of the RL policies available in Go2Py:

```bash
conda create --name rl-deploy python==3.8.10
conda activate rl-deploy
```

Finally, install the deep learning libraries needed for the deployments:

#### DL Frameworks
Download and install appropriate version as described [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048):

```bash
cd ~
wget https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
python -m pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
# Install warp-lang if required
pip install https://github.com/NVIDIA/warp/releases/download/v1.5.1/warp_lang-1.5.1+cu11-py3-none-manylinux2014_aarch64.whl
```
If successful, the following check should return true:

```python
import torch
print(torch.cuda.is_available())
```


### Robot
Now tell the computer on the robot to use the internet shared by the host computer. SSH into the robot's computer with IP address `192.168.123.18`, username `unitree`, and password `123`. Note that the host computer's IP range should have already been set to static mode with an IP in the `192.168.123.x` range where x is anything except IPs already used by the others (e.g. `.18`).

```bash
sudo ip route add default via <host computer IP address>
```

Finally, configure the DNS server by adding the following to the `/etc/resolv.conf` file:
```bash
nameserver 8.8.8.8
```
**Note:** Similar to the host computer, you can save this configuration using the `iptables-persistent` tool.

If everything has been successful, you should be able to access the internet on the robot. Run `ping 8.8.8.8` and `ping www.google.com` to verify this. 

## Installing the Docker
All the deployed services in Go2Py are based on docker so make sure that docker and Nvidia runtime is installed. For this, you can follow through the Isaac-ROS installation instructions [here](https://nvidia-isaac-ros.github.io/getting_started/hardware_setup/compute/jetson_storage.html).  

**Note:** For the LiDAR node, the `nvidia contaner toolkit` should also be installed on the robot.

## Installing the Go2Py

Go2Py has two main components. A set of services running on the robot and a Python package that can run anywhere and is used to communicate with the robot in Python. 

### Installing the Services
SSH into the robot and clone the Go2Py repository:
```bash
git clone https://github.com/Rooholla-KhorramBakht/Go2Py.git
cd Go2Py
```
Go2Py is comprised of the following main services:
- **go2py-bridge.service:** A service that runs the Go2py C++ bridge.
- **go2py-robot-description.service:** A service that subscribes to the `\go2\joint_states` published by the bridge and publishes the robot description and TF2 messages for the sensors and links of the robot. Note that you should add the correct extrinsic parameters of the installed sensors into the robot's xacro file located [here](../deploy/ros2_nodes/go2_description/xacro/robot.xacro) before installing this service. 
- **go2py-hesai.service:** This service runs a customized Lidar driver for the robots that come with the Hesai xt16 Lidars. This customized driver publishes laser_scan messages in addition to the pointcloud data. If your robot does not come with this LiDAR, you don't need to install this service.

The installation for each service is comprised of two commands:
```bash
# For the bridge
make bridge
sudo make bridge_install
# For the robot description service
make robot_description
sudo make robot_description_install
```
The first command for each service makes the corresponding docker image and the second command, copies the service file that runs those images into the appropriate locations of the system and enables them as autorun services. You can check for the success of this installation by checking the output of the `systemctl status service_name.service` command where the `service_name` is replaced with the name of the service you want to check. 

**Note**: At the moment, the implemented LiDAR driver service only supports the XT16 sensor. If you have this sensor on your robot run:
```bash
# For the LiDAR driver
make hesai
sudo make hesai_install
```

### Installing the GoPy 
#### Local Installation
Finally, we need to install the Go2Py Python library on a computer located on the same network as the robot (it could be the onboard computer but can also be any other PCs hooked up to the robot's network). To do so, simply go to the root directory of the Go2Py repository and run:
```bash
pip install -e .
```

After successful installation, open the `Makefile` in the root directory and set the `INTERFACE` variable at the top of the file to the network interface name connected to the robot's network. Then, simply run `make ddscfg` to configure CycloneDDS for proper communication with the robot.

To check the installation, run the interface example [here](../examples/00-robot-interface.ipynb) to make sure you can read the state of the robot. 
#### Using Docker
In addition to local installation, you can use the provided docker support:
##### VSCode Devcontainers
Simply install the devcontainer extension in your VSCode and press `SHIFT+Ctrl+P` and run `Rebuild and Reopen in Container` while you're in the root directory of the project. 
#### Direct

Runing `make docker_start` launces a container with all the requirements installed. Attach to this container and follow the local installation instructions inside the docker.

#### Installation With Isaac-ROS Docker
Running `make isaac_ros_start` starts a docker container with all the required dependencies. The Go2Py repository will be mounted into `/workspaces/Go2Py` and installed in editable mode. Inside this docker environment, you can use the Iasaac-ROS packages.  

Note that our docker image may be extended in a similar way to the Isaac-ROS images (as explained [here](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/index.html)). Simply append the `CONFIG_IMAGE_KEY="ros2_humble.go2py"` [here](../scripts/.isaac_ros_common-config) with the key to your custom Dockerfile name (e.g. `CONFIG_IMAGE_KEY="ros2_humble.go2py.custom"` for `Dockerfile.custom`) and place your docker file under `docker` directory [here](../docker) and append its name with the key you used (e.g. `Dockerfile.custom`). Your custom Dockerfile should start with:
```docker
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
```
In addition to this added docker image layer, you can add your own post execution commands that are excuted each time you start the docker container. These commands should be appended to the end of `workspace-entrypoint.sh` script [here](../docker/scripts/workspace-entrypoint.sh).

#### Enabling GUI Access
In case you're interested in running the simulations inside the docker with GUI support, you need to run `xhost +` on your host terminal to allow X11 requests from the applications inside the Docker.
