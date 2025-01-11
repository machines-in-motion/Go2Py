INTERFACE = enx00e04c006390  # interface name used to talk with the robot

# Target to run the script with the specified interface
ddscfg:
	python3 scripts/set_cyclonedds_config.py $(INTERFACE) && /bin/bash scripts/ros_env_setup.bash

frontcam:
	@cd deploy && docker build --no-cache --tag go2py_frontcam_publisher:latest -f docker/Dockerfile.frontcam .

docker_start:
	@cd .devcontainer && docker compose up go2py

isaac_ros_start:
	@./scripts/run_dev.sh 

nav2:
	@cd deploy && docker build --no-cache --tag go2py_nav2:latest -f docker/Dockerfile.nav2 .

nav2_source:
	@cd deploy && docker build --tag robocaster/navigation2:aarch64 -f docker/Dockerfile.nav2_source .

nav2_start:
	@ ./scripts/run_nav2.sh

mexplore:
	@cd deploy && docker build --no-cache --tag go2py_mexplore:latest -f docker/Dockerfile.mexplore .

mexplore_start:
	@ ./scripts/run_mexplore.sh

messages:
	@cd scripts && ./make_msgs.sh 

realsense:
	@cd deploy/docker && docker build --tag go2py_realsense:latest -f Dockerfile.realsense .

hesai:
	@cd deploy && docker build --no-cache --tag go2py_hesai:latest -f docker/Dockerfile.hesai .

mid360:
	@cd deploy && docker build --tag go2py_mid360:latest -f docker/Dockerfile.mid360 .
	# @cd deploy && docker build --no-cache --tag go2py_mid360:latest -f docker/Dockerfile.mid360 .

bridge:
	@cd deploy && docker build --no-cache --tag go2py_bridge:latest -f docker/Dockerfile.bridge .

robot_description:
	@cd deploy && docker build --no-cache --tag go2py_description:latest -f docker/Dockerfile.robot_description .

frontcam_install:
	@cp deploy/services/go2py-frontcam.service /etc/systemd/system/
	@cp deploy/services/frontcam-v4l-loopback.sh /usr/bin
	@systemctl enable go2py-frontcam.service
	@systemctl start go2py-frontcam.service

hesai_install:
	@cp deploy/services/go2py-hesai.service /etc/systemd/system/
	@systemctl enable go2py-hesai.service
	@systemctl start go2py-hesai.service

mid360_install:
	@cp deploy/services/go2py-mid360.service /etc/systemd/system/
	@systemctl enable go2py-mid360.service
	@systemctl start go2py-mid360.service

bridge_install:
	@cp deploy/services/go2py-bridge.service /etc/systemd/system/
	@systemctl enable go2py-bridge.service
	@systemctl start go2py-bridge.service

robot_description_install:
	@cp deploy/services/go2py-robot-description.service /etc/systemd/system/
	@systemctl enable go2py-robot-description.service
	@systemctl start go2py-robot-description.service

frontcam_uninstall:
	@systemctl disable go2py-frontcam.service
	@systemctl stop go2py-frontcam.service
	@rm /etc/systemd/system/go2py-frontcam.service

hesai_uninstall:
	@systemctl disable go2py-hesai.service
	@systemctl stop go2py-hesai.service
	@rm /etc/systemd/system/go2py-hesai.service

mid360_uninstall:
	@systemctl disable go2py-mid360.service
	@systemctl stop go2py-mid360.service
	@rm /etc/systemd/system/go2py-mid360.service

bridge_uninstall:
	@systemctl disable go2py-bridge.service
	@systemctl stop go2py-bridge.service
	@rm /etc/systemd/system/go2py-bridge.service

robot_description_uninstall:
	@systemctl disable go2py-robot-description.service
	@systemctl stop go2py-robot-description.service
	@rm /etc/systemd/system/go2py-robot-description.service
