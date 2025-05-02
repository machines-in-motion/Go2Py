from dds_telemetry import Go2DDSClient
import time
client = Go2DDSClient(robot_name='go2')

while True:
    odom_state = client.getLegOdom()
    print(odom_state)
    time.sleep(0.1)